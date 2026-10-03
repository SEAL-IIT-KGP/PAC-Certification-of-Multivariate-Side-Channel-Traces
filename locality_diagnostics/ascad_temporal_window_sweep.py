#!/usr/bin/env python3
"""
Tau-sweep experiment for ASCAD desync=0 temporal-window diagnostics.

For each tau, evaluates the pretrained CNN on tau-length windows
(kept at their original time positions, zeros elsewhere) and records the
window success statistics used by the current KL-binomial plotting script.

Checkpoints after every (tau, seed) pair.  Resumable on restart.
"""

import numpy as np
import h5py
import math
import argparse
from pathlib import Path


# ===================================================================
# AES S-box (needed for label computation)
# ===================================================================

# fmt: off
AES_SBOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
]
# fmt: on


# ===================================================================
# Window-success certificate helper
# ===================================================================


def compute_bi_certificate(holdout_success, m, M=1, delta=1e-6, n_classes=256):
    K = n_classes
    margin = math.sqrt(math.log(M / delta) / (2 * m))
    certified_success = holdout_success - margin
    certified_advantage = certified_success - (1.0 / K)
    if certified_advantage <= 0:
        bi_bits = 0.0
    else:
        bi_bits = min(math.log2(1.0 + K * certified_advantage), math.log2(K))
    return {
        "bi_bits": bi_bits,
        "margin": margin,
        "observed_success": holdout_success,
        "certified_success": certified_success,
        "certified_advantage": certified_advantage,
        "n_holdout": m,
        "M": M,
        "delta": delta,
        "n_classes": K,
    }


# ===================================================================
# ASCAD data loading
# ===================================================================


def load_ascad(ascad_h5_path, target_byte=2):
    print(f"Loading ASCAD from: {ascad_h5_path}")
    with h5py.File(ascad_h5_path, "r") as f:
        prof_traces = np.array(f["Profiling_traces/traces"])
        prof_meta = f["Profiling_traces/metadata"]
        prof_plaintext = np.array(prof_meta["plaintext"])
        prof_key = np.array(prof_meta["key"])
        atk_traces = np.array(f["Attack_traces/traces"])
        atk_meta = f["Attack_traces/metadata"]
        atk_plaintext = np.array(atk_meta["plaintext"])
        atk_key = np.array(atk_meta["key"])

    prof_labels = np.array(
        [
            AES_SBOX[int(prof_plaintext[i][target_byte]) ^ int(prof_key[i][target_byte])]
            for i in range(len(prof_traces))
        ],
        dtype=np.int64,
    )
    atk_labels = np.array(
        [AES_SBOX[int(atk_plaintext[i][target_byte]) ^ int(atk_key[i][target_byte])] for i in range(len(atk_traces))],
        dtype=np.int64,
    )

    traces = np.concatenate([prof_traces, atk_traces], axis=0)
    labels = np.concatenate([prof_labels, atk_labels], axis=0)
    print(f"  Profiling: {prof_traces.shape}  Attack: {atk_traces.shape}")
    print(f"  Combined:  {traces.shape}  Labels: {labels.shape}")
    print(f"  Label range: [{labels.min()}, {labels.max()}]")
    print(f"  Unique labels: {len(np.unique(labels))}")
    return traces, labels


# ===================================================================
# Model loading
# ===================================================================


def load_keras_model(model_path):
    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except ImportError:
        pass
    try:
        import keras

        model = keras.models.load_model(model_path, compile=False)
        return model
    except ImportError:
        raise ImportError("Neither tensorflow nor keras is installed.")


# ===================================================================
# Prediction
# ===================================================================


def predict_with_model(model, X, batch_size=1024):
    """Return hard class labels (same as BI script)."""
    expected = tuple(d for d in model.input_shape[1:] if d is not None)

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


# ===================================================================
# Tau-windowing helpers
# ===================================================================


def extract_tau_window(traces, tau, start_pos):
    """Keep the tau-length window starting at start_pos at its original time
    positions and zero all samples outside it (the pretrained CNN expects
    full-length aligned traces)."""
    X = np.zeros_like(traces)
    X[:, start_pos : start_pos + tau] = traces[:, start_pos : start_pos + tau]
    return X


def compute_position_grid(T, tau, strategy="grid"):
    """Compute candidate start positions for a given tau."""
    max_start = T - tau
    if tau >= T:
        return [0]
    if strategy == "center":
        return [max_start // 2]
    elif strategy == "grid":
        stride = max(1, tau // 2)
        positions = list(range(0, max_start + 1, stride))
        if positions[-1] != max_start:
            positions.append(max_start)
        return positions
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def select_best_position(model, traces_prof, labels_prof, tau, strategy="grid", n_subsample=5000, batch_size=1024):
    """Sweep positions on profiling data, return best position + accuracy."""
    T = traces_prof.shape[1]

    if tau >= T:
        # Full trace — no windowing
        y_pred = predict_with_model(model, traces_prof[:n_subsample], batch_size)
        acc = float(np.mean(y_pred == labels_prof[:n_subsample]))
        return 0, acc, {0: acc}

    positions = compute_position_grid(T, tau, strategy)

    if n_subsample < len(labels_prof):
        idx = np.random.choice(len(labels_prof), n_subsample, replace=False)
        X_sub = traces_prof[idx]
        y_sub = labels_prof[idx]
    else:
        X_sub = traces_prof
        y_sub = labels_prof

    best_pos = positions[0]
    best_acc = -1.0
    pos_accs = {}

    for pos in positions:
        X_win = extract_tau_window(X_sub, tau, pos)
        y_pred = predict_with_model(model, X_win, batch_size)
        acc = float(np.mean(y_pred == y_sub))
        pos_accs[pos] = acc
        if acc > best_acc:
            best_acc = acc
            best_pos = pos

    return best_pos, best_acc, pos_accs


def evaluate_tau(model, traces_holdout, labels_holdout, tau, best_pos, batch_size=1024):
    """Evaluate at fixed position on holdout, return success rate."""
    T = traces_holdout.shape[1]
    if tau >= T:
        X_eval = traces_holdout
    else:
        X_eval = extract_tau_window(traces_holdout, tau, best_pos)
    y_pred = predict_with_model(model, X_eval, batch_size)
    return float(np.mean(y_pred == labels_holdout))


# ===================================================================
# Checkpoint: plain text CSV (no csv module needed)
# ===================================================================

HEADER = "tau,seed,best_position,profiling_accuracy,holdout_success,margin,certified_success,certified_advantage,bi_bits,n_holdout,n_profiling,n_positions_tested"


def load_checkpoint(csv_path):
    """Load existing CSV checkpoint. Returns (list of dicts, set of (tau,seed))."""
    results = []
    completed = set()
    p = Path(csv_path)
    if not p.is_file():
        return results, completed
    print(f"Loading checkpoint from: {csv_path}")
    with open(csv_path, "r") as f:
        f.readline()  # header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            r = {
                "tau": int(parts[0]),
                "seed": int(parts[1]),
                "best_position": int(parts[2]),
                "profiling_accuracy": float(parts[3]),
                "holdout_success": float(parts[4]),
                "margin": float(parts[5]),
                "certified_success": float(parts[6]),
                "certified_advantage": float(parts[7]),
                "bi_bits": float(parts[8]),
                "n_holdout": int(parts[9]),
                "n_profiling": int(parts[10]),
                "n_positions_tested": int(parts[11]),
            }
            results.append(r)
            completed.add((r["tau"], r["seed"]))
    print(f"  Loaded {len(results)} completed pairs")
    return results, completed


def append_result(csv_path, result):
    """Append one result row to CSV. Creates file + header if needed."""
    p = Path(csv_path)
    write_header = not p.is_file()
    with open(csv_path, "a") as f:
        if write_header:
            f.write(HEADER + "\n")
        f.write(
            f"{result['tau']},{result['seed']},{result['best_position']},"
            f"{result['profiling_accuracy']:.6f},{result['holdout_success']:.6f},"
            f"{result['margin']:.6f},{result['certified_success']:.6f},"
            f"{result['certified_advantage']:.6f},{result['bi_bits']:.6f},"
            f"{result['n_holdout']},{result['n_profiling']},"
            f"{result['n_positions_tested']}\n"
        )
        f.flush()


def save_summary(results, output_dir):
    """Write per-tau summary CSV."""
    if not results:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "tau_sweep_summary.csv"

    tau_values = sorted(set(r["tau"] for r in results))
    with open(summary_path, "w") as f:
        f.write("tau,bi_mean,bi_std,bi_min,bi_max,success_mean,success_std,success_min,success_max,n_seeds\n")
        for tau in tau_values:
            bi = [r["bi_bits"] for r in results if r["tau"] == tau]
            succ = [r["holdout_success"] for r in results if r["tau"] == tau]
            f.write(
                f"{tau},{np.mean(bi):.6f},{np.std(bi):.6f},"
                f"{np.min(bi):.6f},{np.max(bi):.6f},"
                f"{np.mean(succ):.6f},{np.std(succ):.6f},"
                f"{np.min(succ):.6f},{np.max(succ):.6f},"
                f"{len(bi)}\n"
            )
    print(f"  Summary saved to: {summary_path}")


# ===================================================================
# Plotting (deferred import — only if matplotlib available)
# ===================================================================


def try_plot(results, output_dir):
    """Try to generate plots. If matplotlib is not available, skip silently."""
    if not results:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  matplotlib not available, skipping plots")
        return

    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    tau_values = sorted(set(r["tau"] for r in results))
    bi_means = []
    bi_stds = []
    succ_means = []
    succ_stds = []
    margins = []
    for tau in tau_values:
        bi = [r["bi_bits"] for r in results if r["tau"] == tau]
        succ = [r["holdout_success"] for r in results if r["tau"] == tau]
        marg = [r["margin"] for r in results if r["tau"] == tau]
        bi_means.append(np.mean(bi))
        bi_stds.append(np.std(bi))
        succ_means.append(np.mean(succ))
        succ_stds.append(np.std(succ))
        margins.append(np.mean(marg))

    bi_means = np.array(bi_means)
    bi_stds = np.array(bi_stds)
    succ_means = np.array(succ_means)
    succ_stds = np.array(succ_stds)
    margins = np.array(margins)

    # Plot 1: BI(tau)
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(
            tau_values,
            bi_means,
            yerr=bi_stds,
            fmt="o-",
            color="#e74c3c",
            linewidth=2.5,
            markersize=8,
            capsize=5,
            capthick=2,
            label="BI(tau)",
        )
        if len(tau_values) > 1:
            ax.axhline(
                y=bi_means[-1],
                color="gray",
                linestyle=":",
                alpha=0.5,
                label=f"BI(tau={tau_values[-1]}) = {bi_means[-1]:.2f} bits",
            )
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.2)
        ax.set_xlabel("Temporal window size tau", fontsize=16)
        ax.set_ylabel("Bounded Information (bits)", fontsize=16)
        ax.set_title("BI(tau) - Tau-Local Adversary Certification\n(ASCAD desync=0, pretrained CNN)", fontsize=18)
        ax.set_xscale("log")
        ax.set_xticks(tau_values)
        ax.set_xticklabels([str(t) for t in tau_values], fontsize=12)
        ax.tick_params(axis="y", labelsize=14)
        ax.legend(fontsize=13, loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "tau_sweep_bi.pdf", dpi=300)
        plt.savefig(figures_dir / "tau_sweep_bi.png", dpi=150)
        plt.close()
        print(f"  Saved: {figures_dir / 'tau_sweep_bi.pdf'}")
    except Exception as e:
        print(f"  Plot 1 (BI) failed: {e}")

    # Plot 2: Success rate(tau)
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(
            tau_values,
            succ_means,
            yerr=succ_stds,
            fmt="s-",
            color="#3498db",
            linewidth=2.5,
            markersize=8,
            capsize=5,
            capthick=2,
            label="Holdout success rate",
        )
        ax.fill_between(
            tau_values,
            succ_means - margins,
            succ_means + margins,
            alpha=0.15,
            color="#3498db",
            label="confidence margin",
        )
        ax.axhline(y=1 / 256, color="gray", linestyle=":", alpha=0.5, label=f"Random guess (1/256 = {1 / 256:.4f})")
        ax.set_xlabel("Temporal window size tau", fontsize=16)
        ax.set_ylabel("Holdout success rate", fontsize=16)
        ax.set_title("Success Rate vs tau\n(ASCAD desync=0, pretrained CNN)", fontsize=18)
        ax.set_xscale("log")
        ax.set_xticks(tau_values)
        ax.set_xticklabels([str(t) for t in tau_values], fontsize=12)
        ax.tick_params(axis="y", labelsize=14)
        ax.legend(fontsize=13, loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "tau_sweep_success.pdf", dpi=300)
        plt.savefig(figures_dir / "tau_sweep_success.png", dpi=150)
        plt.close()
        print(f"  Saved: {figures_dir / 'tau_sweep_success.pdf'}")
    except Exception as e:
        print(f"  Plot 2 (success) failed: {e}")

    # Plot 3: Best position per tau
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        for seed in sorted(set(r["seed"] for r in results)):
            taus_s = [r["tau"] for r in results if r["seed"] == seed]
            pos_s = [r["best_position"] for r in results if r["seed"] == seed]
            ax.plot(taus_s, pos_s, "o--", alpha=0.5, markersize=6, label=f"Seed {seed}")
        ax.set_xlabel("Temporal window size tau", fontsize=16)
        ax.set_ylabel("Best window start position", fontsize=16)
        ax.set_title("Selected Window Position vs tau\n(ASCAD desync=0)", fontsize=18)
        ax.set_xscale("log")
        ax.set_xticks(tau_values)
        ax.set_xticklabels([str(t) for t in tau_values], fontsize=12)
        ax.tick_params(axis="y", labelsize=14)
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "tau_sweep_positions.pdf", dpi=300)
        plt.savefig(figures_dir / "tau_sweep_positions.png", dpi=150)
        plt.close()
        print(f"  Saved: {figures_dir / 'tau_sweep_positions.pdf'}")
    except Exception as e:
        print(f"  Plot 3 (positions) failed: {e}")


# ===================================================================
# Print summary table
# ===================================================================


def print_summary(results):
    if not results:
        return
    tau_values = sorted(set(r["tau"] for r in results))

    print("")
    print("=" * 80)
    print("TAU-SWEEP SUMMARY")
    print("=" * 80)
    print(
        f"  {'tau':>6s}  {'BI mean':>10s}  {'BI std':>9s}  "
        f"{'Success':>10s}  {'Succ std':>9s}  {'Seeds':>6s}  {'Positions':>10s}"
    )
    print(f"  {'-' * 6}  {'-' * 10}  {'-' * 9}  {'-' * 10}  {'-' * 9}  {'-' * 6}  {'-' * 10}")

    prev_bi = -1.0
    monotonic = True
    for tau in tau_values:
        bi = [r["bi_bits"] for r in results if r["tau"] == tau]
        succ = [r["holdout_success"] for r in results if r["tau"] == tau]
        npos = [r["n_positions_tested"] for r in results if r["tau"] == tau][0]
        bi_mean = np.mean(bi)
        arrow = ""
        if bi_mean < prev_bi - 1e-6:
            arrow = " <-- NOT MONOTONIC"
            monotonic = False
        prev_bi = bi_mean
        print(
            f"  {tau:6d}  {bi_mean:10.4f}  {np.std(bi):9.4f}  "
            f"{np.mean(succ):10.6f}  {np.std(succ):9.6f}  {len(bi):6d}  {npos:10d}{arrow}"
        )

    print(f"\n  Monotonicity check: {'PASSED' if monotonic else 'FAILED'}")
    print(f"  Random guess: {1 / 256:.6f}")


# ===================================================================
# Main
# ===================================================================


def main():
    print("=" * 70)
    print("Tau-Sweep BI Experiment: ASCAD desync=0")
    print("=" * 70)

    parser = argparse.ArgumentParser(description="Tau-sweep BI experiment for ASCAD desync=0")
    parser.add_argument("--ascad-h5", type=str, required=True, help="Path to ASCAD.h5 (desync=0 dataset)")
    parser.add_argument("--cnn-model", type=str, required=True, help="Path to pretrained CNN .h5 file")
    parser.add_argument("--target-byte", type=int, default=2)
    parser.add_argument(
        "--tau-values", type=str, default="5,10,20,50,100,200,350,500,700", help="Comma-separated tau values"
    )
    parser.add_argument("--strategy", type=str, default="grid", choices=["grid", "center"])
    parser.add_argument("--holdout-size", type=int, default=25000)
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--n-classes", type=int, default=256)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-subsample", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output-dir", type=str, default="results/tau_sweep")

    args = parser.parse_args()

    tau_values = sorted([int(t.strip()) for t in args.tau_values.split(",")])

    print(f"  CNN model:    {args.cnn_model}")
    print(f"  ASCAD data:   {args.ascad_h5}")
    print(f"  Tau values:   {tau_values}")
    print(f"  Strategy:     {args.strategy}")
    print(f"  Holdout size: {args.holdout_size}")
    print(f"  Delta:        {args.delta}")
    print(f"  Seeds:        {args.n_seeds}")
    print(f"  Subsample:    {args.n_subsample}")
    print(f"  Output:       {args.output_dir}")
    print("")

    # Create output dirs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    csv_path = str(output_dir / "tau_sweep_results.csv")

    # Load checkpoint
    results, completed = load_checkpoint(csv_path)
    if completed:
        print(f"\nRESUMING: found {len(completed)} completed (tau, seed) pairs")
        for tau in tau_values:
            done = sum(1 for t, s in completed if t == tau)
            print(f"  tau={tau:4d}: {done}/{args.n_seeds} seeds done")
    else:
        print("\nStarting fresh (no checkpoint found)")

    total_pairs = len(tau_values) * args.n_seeds
    remaining = total_pairs - len(completed)
    print(f"\nTotal pairs: {total_pairs},  Remaining: {remaining}")

    if remaining == 0:
        print("\nAll pairs already completed! Regenerating outputs...")
        print_summary(results)
        save_summary(results, args.output_dir)
        try_plot(results, args.output_dir)
        print(f"\n{'=' * 70}")
        print("Done (nothing new to compute).")
        print(f"{'=' * 70}")
        return

    # Load data
    traces, labels = load_ascad(args.ascad_h5, target_byte=args.target_byte)

    # Load CNN
    print("\nLoading CNN model...")
    model = load_keras_model(args.cnn_model)
    print(f"  Input shape:  {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")

    T = traces.shape[1]
    n_total = len(labels)
    count_new = 0

    print(f"\nTrace length T = {T}")
    print(f"Total traces:   {n_total}")

    # Main loop: seed -> tau
    for seed in range(args.n_seeds):
        np.random.seed(seed)
        indices = np.random.permutation(n_total)

        n_holdout = min(args.holdout_size, n_total // 2)
        holdout_idx = indices[:n_holdout]
        prof_idx = indices[n_holdout:]

        X_holdout = traces[holdout_idx]
        y_holdout = labels[holdout_idx]
        X_prof = traces[prof_idx]
        y_prof = labels[prof_idx]

        # Check if any tau remains for this seed
        seed_remaining = [tau for tau in tau_values if (tau, seed) not in completed]
        if not seed_remaining:
            continue

        print(f"\n{'=' * 70}")
        print(f"Seed {seed}: profiling={len(y_prof)}, holdout={n_holdout}")
        print(f"  Remaining taus: {seed_remaining}")
        print(f"{'=' * 70}")

        for tau in tau_values:
            if (tau, seed) in completed:
                continue

            n_positions = len(compute_position_grid(T, tau, args.strategy))
            print(f"  tau={tau:4d}  positions={n_positions:4d}  ", end="", flush=True)

            # Phase 1: position selection on profiling data
            best_pos, prof_acc, pos_accs = select_best_position(
                model, X_prof, y_prof, tau, args.strategy, args.n_subsample, args.batch_size
            )

            # Phase 2: holdout evaluation
            holdout_success = evaluate_tau(model, X_holdout, y_holdout, tau, best_pos, args.batch_size)

            # Phase 3: BI certificate (M=1)
            cert = compute_bi_certificate(
                holdout_success=holdout_success, m=n_holdout, M=1, delta=args.delta, n_classes=args.n_classes
            )

            prof_acc_str = f"{prof_acc:.4f}" if prof_acc is not None else "N/A"
            print(
                f"pos={best_pos:4d}  prof={prof_acc_str}  "
                f"holdout={holdout_success:.6f}  "
                f"margin={cert['margin']:.6f}  "
                f"BI={cert['bi_bits']:.4f} bits"
            )

            result = {
                "tau": tau,
                "seed": seed,
                "best_position": best_pos,
                "profiling_accuracy": prof_acc if prof_acc is not None else holdout_success,
                "holdout_success": holdout_success,
                "margin": cert["margin"],
                "certified_success": cert["certified_success"],
                "certified_advantage": cert["certified_advantage"],
                "bi_bits": cert["bi_bits"],
                "n_holdout": n_holdout,
                "n_profiling": len(y_prof),
                "n_positions_tested": n_positions,
            }

            # Checkpoint: append immediately
            results.append(result)
            completed.add((tau, seed))
            append_result(csv_path, result)
            count_new += 1

        # After each seed: save summary + try plots
        print(f"\n  [Checkpoint] Saving after seed {seed}...")
        save_summary(results, args.output_dir)
        try_plot(results, args.output_dir)
        print_summary(results)

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"COMPLETE: {count_new} new pairs computed, {len(results)} total")
    print(f"{'=' * 70}")
    save_summary(results, args.output_dir)
    try_plot(results, args.output_dir)
    print_summary(results)

    print(f"\n  Results CSV:   {csv_path}")
    print(f"  Summary CSV:   {output_dir / 'tau_sweep_summary.csv'}")
    print(f"  Figures:       {output_dir / 'figures/'}")

    print(f"\n{'=' * 70}")
    print("Done.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
