"""
Bounded Information (BI) Certificate Computation

This module implements the core BI certification framework used in the current
experiments:
- KL-binomial confidence endpoints for exact-recovery success rates.
- Union-bound correction over a declared attack suite.
- Conversion from certified guessing probability to BI in bits.

For a suite of M attackers with n_te attack traces, the upper endpoint p_plus
is the largest success probability whose Bernoulli KL distance from the
observed success rate is within log(M/delta)/n_te. BI is then

    BI = log2(1 + K * max(p_plus - 1/K, 0)).
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class BICertificate:
    """Container for BI certificate results following Definition bi."""
    bi_bits: float                  # Certified bounded information in bits (Eq. bi-def)
    observed_success: float         # Observed success rate p̂_max on holdout
    certified_success: float        # Upper bound on P_guess^H (p̂_max + margin)
    certified_advantage: float      # Certified advantage ε* = certified_success - 1/K
    margin: float                   # Upward KL-binomial confidence slack
    holdout_size: int               # n_te - number of holdout samples
    suite_size: int                 # M - number of models in attack suite
    delta: float                    # Failure probability
    n_classes: int                  # K - number of classes (e.g., 256 for AES byte)


def bernoulli_kl(q: float, p: float) -> float:
    eps = 1e-15
    q = min(1.0 - eps, max(eps, float(q)))
    p = min(1.0 - eps, max(eps, float(p)))
    return q * np.log(q / p) + (1.0 - q) * np.log((1.0 - q) / (1.0 - p))


def kl_upper_endpoint(successes: int, n: int, alpha: float) -> float:
    """Compute the upper KL-binomial confidence endpoint."""
    if n <= 0:
        raise ValueError("n must be positive")
    successes = int(max(0, min(n, successes)))
    q = successes / n
    if successes >= n:
        return 1.0
    target = np.log(1.0 / alpha) / n
    lo, hi = q, 1.0 - 1e-15
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bernoulli_kl(q, mid) > target:
            hi = mid
        else:
            lo = mid
    return lo


def kl_confidence_slack(observed_success: float, m: int, M: int, delta: float) -> float:
    """Return p_plus - p_hat under a suite-union KL-binomial endpoint."""
    alpha = delta / max(1, int(M))
    successes = int(round(float(observed_success) * int(m)))
    p_hat = successes / int(m)
    return kl_upper_endpoint(successes, int(m), alpha) - p_hat


def advantage_to_bi_bits(epsilon: float, n_classes: int = 256) -> float:
    """
    Convert certified advantage to bounded information in bits (Eq. bi-def).

    From Definition bi (Section 3):
        BI(D,H,δ) = min{β(ε*, K), log₂(K)} = min{log₂(1 + K·ε*), log₂(K)}

    where β(ε, K) = log₂(1 + K·ε) from Theorem guess-to-mi.

    Args:
        epsilon: Certified advantage ε* (must be >= 0)
        n_classes: Number of classes K (default 256 for AES byte)

    Returns:
        Bounded information in bits
    """
    # Clamp advantage to valid range [0, (K-1)/K]
    epsilon = max(0.0, epsilon)
    epsilon = min(epsilon, (n_classes - 1) / n_classes)

    # β(ε, K) = log₂(1 + K·ε)
    beta = np.log2(1.0 + n_classes * epsilon)

    # BI = min{β(ε*, K), log₂(K)}
    max_bi = np.log2(n_classes)
    bi = min(beta, max_bi)

    return max(0.0, bi)  # BI is non-negative


def success_to_bi_bits(p: float, n_classes: int = 256) -> float:
    """
    Convert success probability to bounded information in bits.

    This is equivalent to advantage_to_bi_bits with ε = p - 1/K.

    From Theorem guess-to-mi (Eq. mi-guess-bound):
        I(X;Y) ≤ log₂(K · p_g)

    For P_guess = 1/K + ε, this gives:
        BI = log₂(K · (1/K + ε)) = log₂(1 + K·ε)

    Equivalently: BI = log₂(K · p) where p is success probability.

    For a 256-class problem (AES byte):
        - Random guess: p = 1/256 → BI = 0 bits
        - Perfect: p = 1 → BI = 8 bits

    Args:
        p: Success probability (must be >= 1/n_classes)
        n_classes: Number of classes K (default 256 for AES byte)

    Returns:
        Bounded information in bits
    """
    # Clamp to valid range [1/K, 1]
    p_min = 1.0 / n_classes
    p = max(p, p_min)
    p = min(p, 1.0)

    # Convert to advantage and use the canonical formula
    epsilon = p - p_min  # ε = p - 1/K
    return advantage_to_bi_bits(epsilon, n_classes)


def compute_bi_certificate(
    holdout_success: float,
    m: int,
    M: int = 1,
    delta: float = 1e-6,
    n_classes: int = 256
) -> BICertificate:
    """
    Compute the BI certificate for an observed success rate (Section 3 methodology).

    This is the core certification function implementing the current
    KL-binomial suite endpoint:

    1. Convert the observed success rate to an integer success count.

    2. Compute the KL-binomial upper endpoint with alpha = delta / M.

    3. Convert p_plus - 1/K to BI bits.


    Args:
        holdout_success: Observed success rate p̂_max on holdout set
        m: Holdout set size (n_te in paper)
        M: Number of models in attack suite (default 1 for single attack)
        delta: Failure probability (default 1e-6, confidence = 1-δ)
        n_classes: Number of target classes K (default 256 for AES byte)

    Returns:
        BICertificate containing all certification details
    """
    successes = int(round(float(holdout_success) * int(m)))
    successes = max(0, min(int(m), successes))
    observed_success = successes / int(m)

    alpha = float(delta) / max(1, int(M))
    certified_success = kl_upper_endpoint(successes, int(m), alpha)
    margin = certified_success - observed_success

    baseline = 1.0 / n_classes
    certified_advantage = certified_success - baseline

    bi_bits = advantage_to_bi_bits(certified_advantage, n_classes)

    return BICertificate(
        bi_bits=bi_bits,
        observed_success=observed_success,
        certified_success=certified_success,
        certified_advantage=certified_advantage,
        margin=margin,
        holdout_size=m,
        suite_size=M,
        delta=delta,
        n_classes=n_classes
    )


def compute_bi_from_predictions(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    M: int = 1,
    delta: float = 1e-6,
    n_classes: int = 256
) -> BICertificate:
    """
    Compute BI certificate directly from predictions and labels.

    Convenience function that computes p̂_succ from predictions and calls
    the core certification function.

    Args:
        y_pred: Predicted class labels (argmax of model output)
        y_true: True class labels
        M: Number of models in attack suite (1 for single attack)
        delta: Failure probability
        n_classes: Number of target classes K

    Returns:
        BICertificate with all certification details
    """
    m = len(y_true)
    observed_success = np.mean(y_pred == y_true)

    return compute_bi_certificate(
        holdout_success=observed_success,
        m=m,
        M=M,
        delta=delta,
        n_classes=n_classes
    )


def compute_bi_from_probabilities(
    y_pred_proba: np.ndarray,
    y_true: np.ndarray,
    M: int = 1,
    delta: float = 1e-6
) -> BICertificate:
    """
    Compute BI certificate from probability predictions.

    Args:
        y_pred_proba: Predicted probabilities (n_samples, n_classes)
        y_true: True class labels
        M: Number of models in attack suite
        delta: Failure probability

    Returns:
        BICertificate
    """
    n_classes = y_pred_proba.shape[1]
    y_pred = np.argmax(y_pred_proba, axis=1)

    return compute_bi_from_predictions(
        y_pred=y_pred,
        y_true=y_true,
        M=M,
        delta=delta,
        n_classes=n_classes
    )


def compute_attack_suite_bi(
    models: List,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    delta: float = 1e-6,
    n_classes: int = 256
) -> Tuple[BICertificate, int]:
    """
    Compute BI certificate for an attack suite (multiple models).

    Selects the best model by holdout success and applies union bound
    correction for trying M models.

    Args:
        models: List of trained models with .predict() method
        X_holdout: Holdout features
        y_holdout: Holdout labels
        delta: Failure probability
        n_classes: Number of target classes

    Returns:
        Tuple of (BICertificate for best model, index of best model)
    """
    M = len(models)
    m = len(y_holdout)

    best_success = 0.0
    best_idx = 0

    for idx, model in enumerate(models):
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_holdout)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_holdout)
            if hasattr(y_pred, 'argmax'):
                y_pred = np.argmax(y_pred, axis=1)

        success = np.mean(y_pred == y_holdout)

        if success > best_success:
            best_success = success
            best_idx = idx

    # Compute certificate with union bound correction
    certificate = compute_bi_certificate(
        holdout_success=best_success,
        m=m,
        M=M,
        delta=delta,
        n_classes=n_classes
    )

    return certificate, best_idx


def bi_scaling_analysis(
    observed_success: float,
    m_values: List[int],
    M_values: List[int],
    delta_values: List[float],
    n_classes: int = 256
) -> Dict:
    """
    Analyze how BI certificate scales with m, M, and delta.

    Useful for non-vacuity and scaling analysis.

    Args:
        observed_success: Fixed observed success rate
        m_values: List of holdout sizes to evaluate
        M_values: List of suite sizes to evaluate
        delta_values: List of confidence levels to evaluate
        n_classes: Number of target classes

    Returns:
        Dictionary with scaling results
    """
    results = {
        'm': [],
        'M': [],
        'delta': [],
        'bi_bits': [],
        'margin': [],
        'certified_success': []
    }

    for m in m_values:
        for M in M_values:
            for delta in delta_values:
                cert = compute_bi_certificate(
                    holdout_success=observed_success,
                    m=m,
                    M=M,
                    delta=delta,
                    n_classes=n_classes
                )

                results['m'].append(m)
                results['M'].append(M)
                results['delta'].append(delta)
                results['bi_bits'].append(cert.bi_bits)
                results['margin'].append(cert.margin)
                results['certified_success'].append(cert.certified_success)

    return results


# Convenience functions for common configurations
def quick_bi(success: float, m: int, M: int = 1) -> float:
    """Quick BI computation with default delta=1e-6 for 256 classes."""
    return compute_bi_certificate(success, m, M).bi_bits


def margin_only(m: int, M: int = 1, delta: float = 1e-6) -> float:
    """Compute KL-binomial slack at p_hat=0.5 for a quick scale estimate."""
    return kl_confidence_slack(0.5, m, M, delta)


def min_holdout_size(eta: float, M: int = 1, delta: float = 1e-6) -> int:
    """
    Approximate minimum holdout size for target confidence slack eta.

    This conservative approximation is useful for rough planning before the
    observed success rate is known.

    Args:
        eta: Target maximum confidence margin
        M: Number of models in attack suite
        delta: Failure probability

    Returns:
        Minimum holdout size (ceiling)
    """
    return int(np.ceil(np.log(M / delta) / (2 * eta ** 2)))


def tightness_diagnostic(cert: BICertificate) -> Dict:
    """
    Compute tightness diagnostic for a BI certificate (Section 3.4).

    Reports the gap between observed and certified values, which quantifies
    the statistical slack of the certificate.

    Args:
        cert: BICertificate object

    Returns:
        Dictionary with diagnostic values
    """
    observed_advantage = cert.observed_success - (1.0 / cert.n_classes)

    return {
        'observed_success': cert.observed_success,
        'certified_success': cert.certified_success,
        'success_slack': cert.certified_success - cert.observed_success,
        'observed_advantage': observed_advantage,
        'certified_advantage': cert.certified_advantage,
        'advantage_slack': cert.certified_advantage - observed_advantage,
        'margin': cert.margin,
        'bi_bits': cert.bi_bits,
        'observed_bi_bits': success_to_bi_bits(cert.observed_success, cert.n_classes),
        'bi_slack': cert.bi_bits - success_to_bi_bits(cert.observed_success, cert.n_classes),
    }


if __name__ == "__main__":
    # Example usage and validation matching the current KL-binomial convention.
    print("BI Certificate Module - Example Usage")
    print("=" * 60)
    print("  Suite endpoint: Bernoulli-KL upper confidence endpoint with alpha=delta/M")
    print("  BI = log₂(1 + K·max(p_plus - 1/K, 0))")
    print("=" * 60)

    # Example 1: Simple certificate computation
    cert = compute_bi_certificate(
        holdout_success=0.15,  # 15% success (p̂_max)
        m=50000,               # 50k holdout (n_te)
        M=10,                  # 10 models tried
        delta=1e-6             # 99.9999% confidence
    )

    print(f"\nExample 1: Suite certification (M=10 models)")
    print(f"  Observed success (p̂_max): {cert.observed_success:.4f}")
    print(f"  KL-binomial slack: {cert.margin:.4f}")
    print(f"  Certified success p_plus: {cert.certified_success:.4f}")
    print(f"  Certified advantage (ε*): {cert.certified_advantage:.4f}")
    print(f"  BI = log₂(1 + K·ε*): {cert.bi_bits:.3f} bits")

    # Example 2: Scaling with holdout size
    print(f"\nExample 2: Scaling with holdout size n_te (M=10, δ=1e-6)")
    for m in [5000, 10000, 50000, 100000]:
        cert = compute_bi_certificate(0.15, m, M=10, delta=1e-6)
        print(f"  n_te={m:6d}: slack={cert.margin:.4f}, ε*={cert.certified_advantage:.4f}, BI={cert.bi_bits:.3f} bits")

    # Example 3: Scaling with suite size M
    print(f"\nExample 3: Scaling with suite size M (logarithmic penalty)")
    for M in [1, 10, 50, 200]:
        cert = compute_bi_certificate(0.15, m=50000, M=M, delta=1e-6)
        print(f"  M={M:3d}: slack={cert.margin:.4f}, ε*={cert.certified_advantage:.4f}, BI={cert.bi_bits:.3f} bits")

    # Example 4: Reference BI values using both formulas
    print(f"\nExample 4: Reference BI values (K=256 classes)")
    print(f"  Using BI = log₂(K·p) = log₂(1 + K·ε) where ε = p - 1/K")
    for p in [1/256, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]:
        epsilon = p - 1/256
        bi_from_p = success_to_bi_bits(p, 256)
        bi_from_eps = advantage_to_bi_bits(epsilon, 256)
        print(f"  p={p:.4f}, ε={epsilon:.4f}: BI={bi_from_p:.3f} bits (both methods agree: {np.isclose(bi_from_p, bi_from_eps)})")

    # Example 5: Verify endpoint helpers
    print(f"\nExample 5: Verify KL-binomial endpoint helpers")
    m, M, delta = 50000, 10, 1e-6
    margin_single = kl_confidence_slack(0.15, m, 1, delta)
    margin_suite = kl_confidence_slack(0.15, m, M, delta)
    print(f"  Single attack (M=1): slack = {margin_single:.6f}")
    print(f"  Suite attack (M=10): slack = {margin_suite:.6f}")
    print(f"  Ratio suite/single: {margin_suite / margin_single:.3f}")
