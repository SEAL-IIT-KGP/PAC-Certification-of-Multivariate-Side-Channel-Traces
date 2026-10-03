"""EPI and key-rank helpers shared by the selected-model evaluations.

All functions work on an `(n, 256)` matrix of log-probabilities produced by one
frozen model, so single-trace success, EPI, and multi-trace key recovery are
computed from the same predictions.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

CALIBRATION_SEED = 20260908
PERMUTATION_SEED = 20260908
MAX_CALIBRATION_TRACES = 10000
BETA_MAX = 64.0
CAMPAIGNS = 100
TRACE_BUDGETS = [1, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 12500, 25000]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logsumexp(array: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    if not keepdims:
        result = np.squeeze(result, axis=axis)
    return result


def prepare_input(model, traces: np.ndarray) -> np.ndarray:
    """Crop or zero-pad traces to the model's input width and add the channel axis for CNNs."""
    expected = tuple(int(x) for x in model.input_shape[1:] if x is not None)
    x = traces
    if len(expected) == 2 and expected[-1] == 1 and x.ndim == 2:
        width = expected[0]
        if x.shape[1] > width:
            x = x[:, :width]
        elif x.shape[1] < width:
            x = np.pad(x, ((0, 0), (0, width - x.shape[1])))
        x = x[..., None]
    elif len(expected) == 1 and x.ndim == 2:
        width = expected[0]
        if x.shape[1] > width:
            x = x[:, :width]
        elif x.shape[1] < width:
            x = np.pad(x, ((0, 0), (0, width - x.shape[1])))
    return np.asarray(x, dtype=np.float32)


def normalize_logp(raw: np.ndarray) -> np.ndarray:
    """Turn model outputs (probabilities or logits) into normalized log-probabilities."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 256:
        raise AssertionError(f"expected (n, 256) output, got {raw.shape}")
    row_sums = raw.sum(axis=1)
    if np.all(raw >= 0.0) and np.all(np.isfinite(row_sums)) and np.median(np.abs(row_sums - 1.0)) < 1e-3:
        logp = np.log(np.maximum(raw, np.finfo(np.float64).tiny))
    else:
        logp = raw
    return logp - logsumexp(logp, axis=1, keepdims=True)


def predict_logp(model, traces: np.ndarray, batch_size: int) -> np.ndarray:
    raw = model.predict(prepare_input(model, traces), batch_size=batch_size, verbose=0)
    if isinstance(raw, (tuple, list)):
        if len(raw) != 1:
            raise ValueError(f"expected one model output, got {len(raw)}")
        raw = raw[0]
    return normalize_logp(raw)


def information_records(logp: np.ndarray, labels: np.ndarray, beta: float) -> np.ndarray:
    """Per-trace information (bits) of the tempered model p^beta / sum(p^beta); beta = 1 gives raw PI."""
    scaled = float(beta) * logp
    selected = scaled[np.arange(len(labels)), labels]
    normalized = selected - logsumexp(scaled, axis=1)
    return 8.0 + normalized / math.log(2.0)


def fit_beta(logp: np.ndarray, labels: np.ndarray, beta_max: float = BETA_MAX) -> tuple[float, float]:
    """Inverse temperature in [0, beta_max] that maximizes the empirical information on calibration traces."""

    def loss(beta: float) -> float:
        return float(8.0 - np.mean(information_records(logp, labels, beta)))

    result = minimize_scalar(
        loss, bounds=(0.0, float(beta_max)), method="bounded", options={"xatol": 1e-8, "maxiter": 500}
    )
    candidates = [(0.0, loss(0.0)), (float(result.x), float(result.fun)), (float(beta_max), loss(beta_max))]
    beta, value = min(candidates, key=lambda item: item[1])
    if not np.isfinite(value):
        raise FloatingPointError("non-finite calibration loss")
    return beta, value


def calibration_indices(n_profiling: int) -> np.ndarray:
    rng = np.random.default_rng(CALIBRATION_SEED)
    return np.sort(rng.choice(n_profiling, min(n_profiling, MAX_CALIBRATION_TRACES), replace=False))


def key_rank_campaigns(
    logp: np.ndarray,
    targets: np.ndarray,
    key: int,
    budgets: list[int] | None = None,
    repeats: int = CAMPAIGNS,
    seed: int = PERMUTATION_SEED,
) -> tuple[list[dict[str, object]], np.ndarray, list[int]]:
    """Key-byte rank after accumulating log-probabilities over shuffled orders of the fixed attack pool.

    `targets[i, k]` is the intermediate label of trace `i` under key guess `k`.
    Rank 0 means the correct key has the highest score; ties go to the smaller key index.
    """
    n = len(logp)
    budgets = [b for b in (budgets or TRACE_BUDGETS) if b <= n]
    evidence = logp[np.arange(n)[:, None], targets]
    rng = np.random.default_rng(seed)
    all_ranks = []
    for _ in range(repeats):
        # One full shuffle per campaign; only its first max(budgets) traces are accumulated.
        order = rng.permutation(n)[: budgets[-1]]
        scores = np.cumsum(evidence[order], axis=0)[np.array(budgets) - 1]
        true = scores[:, int(key)]
        rank = np.sum(scores > true[:, None], axis=1) + np.sum(
            (scores == true[:, None]) & (np.arange(256) < int(key))[None, :], axis=1
        )
        all_ranks.append(rank)
    ranks = np.asarray(all_ranks)
    rows = [
        {
            "traces": int(budget),
            "campaigns": int(repeats),
            "successes": int(np.sum(ranks[:, j] == 0)),
            "success_rate": float(np.mean(ranks[:, j] == 0)),
            "mean_rank": float(np.mean(ranks[:, j])),
            "median_rank": float(np.median(ranks[:, j])),
        }
        for j, budget in enumerate(budgets)
    ]
    return rows, ranks, budgets
