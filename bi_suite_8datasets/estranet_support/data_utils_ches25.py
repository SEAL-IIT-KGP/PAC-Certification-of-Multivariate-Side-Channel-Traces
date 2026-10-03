"""Data utilities for loading the CHES CTF 2025 dataset.

The dataset structure is similar to ASCAD (HDF5 with Profiling_traces /
Attack_traces groups), but targets byte 0 of the AES S-box output.

Unlike ASCAD, the CHES CTF 2025 .h5 file may not contain a precomputed
'labels' dataset. In that case, labels are computed on the fly as:
    label = AES_SBOX[plaintext[0] XOR key[0]]

By default, traces use the CHES2025-Challenge preprocessing from
LucianoXu/CHES2025-Challenge: optional notch denoising, attack-distribution
mean shift toward profiling traces with key byte 127, then profiling-fitted
per-sample standardization. Set CHES25_PREPROCESS=official to use the simpler
official start-kit StandardScaler, or CHES25_PREPROCESS=none for raw traces.

Reference:
    GE Wars: The Deep Learning SCA Battle, CHES Challenge 2025
    https://pace-tl.gitbook.io/ches-challenge-2025
"""

import numpy as np
import tensorflow as tf
import h5py

import os
import sys

# Target byte index for the CHES CTF 2025 challenge
TARGET_BYTE = 0

# Rijndael AES S-box (used to compute labels from plaintext and key)
AES_SBOX = np.array(
    [
        0x63,
        0x7C,
        0x77,
        0x7B,
        0xF2,
        0x6B,
        0x6F,
        0xC5,
        0x30,
        0x01,
        0x67,
        0x2B,
        0xFE,
        0xD7,
        0xAB,
        0x76,
        0xCA,
        0x82,
        0xC9,
        0x7D,
        0xFA,
        0x59,
        0x47,
        0xF0,
        0xAD,
        0xD4,
        0xA2,
        0xAF,
        0x9C,
        0xA4,
        0x72,
        0xC0,
        0xB7,
        0xFD,
        0x93,
        0x26,
        0x36,
        0x3F,
        0xF7,
        0xCC,
        0x34,
        0xA5,
        0xE5,
        0xF1,
        0x71,
        0xD8,
        0x31,
        0x15,
        0x04,
        0xC7,
        0x23,
        0xC3,
        0x18,
        0x96,
        0x05,
        0x9A,
        0x07,
        0x12,
        0x80,
        0xE2,
        0xEB,
        0x27,
        0xB2,
        0x75,
        0x09,
        0x83,
        0x2C,
        0x1A,
        0x1B,
        0x6E,
        0x5A,
        0xA0,
        0x52,
        0x3B,
        0xD6,
        0xB3,
        0x29,
        0xE3,
        0x2F,
        0x84,
        0x53,
        0xD1,
        0x00,
        0xED,
        0x20,
        0xFC,
        0xB1,
        0x5B,
        0x6A,
        0xCB,
        0xBE,
        0x39,
        0x4A,
        0x4C,
        0x58,
        0xCF,
        0xD0,
        0xEF,
        0xAA,
        0xFB,
        0x43,
        0x4D,
        0x33,
        0x85,
        0x45,
        0xF9,
        0x02,
        0x7F,
        0x50,
        0x3C,
        0x9F,
        0xA8,
        0x51,
        0xA3,
        0x40,
        0x8F,
        0x92,
        0x9D,
        0x38,
        0xF5,
        0xBC,
        0xB6,
        0xDA,
        0x21,
        0x10,
        0xFF,
        0xF3,
        0xD2,
        0xCD,
        0x0C,
        0x13,
        0xEC,
        0x5F,
        0x97,
        0x44,
        0x17,
        0xC4,
        0xA7,
        0x7E,
        0x3D,
        0x64,
        0x5D,
        0x19,
        0x73,
        0x60,
        0x81,
        0x4F,
        0xDC,
        0x22,
        0x2A,
        0x90,
        0x88,
        0x46,
        0xEE,
        0xB8,
        0x14,
        0xDE,
        0x5E,
        0x0B,
        0xDB,
        0xE0,
        0x32,
        0x3A,
        0x0A,
        0x49,
        0x06,
        0x24,
        0x5C,
        0xC2,
        0xD3,
        0xAC,
        0x62,
        0x91,
        0x95,
        0xE4,
        0x79,
        0xE7,
        0xC8,
        0x37,
        0x6D,
        0x8D,
        0xD5,
        0x4E,
        0xA9,
        0x6C,
        0x56,
        0xF4,
        0xEA,
        0x65,
        0x7A,
        0xAE,
        0x08,
        0xBA,
        0x78,
        0x25,
        0x2E,
        0x1C,
        0xA6,
        0xB4,
        0xC6,
        0xE8,
        0xDD,
        0x74,
        0x1F,
        0x4B,
        0xBD,
        0x8B,
        0x8A,
        0x70,
        0x3E,
        0xB5,
        0x66,
        0x48,
        0x03,
        0xF6,
        0x0E,
        0x61,
        0x35,
        0x57,
        0xB9,
        0x86,
        0xC1,
        0x1D,
        0x9E,
        0xE1,
        0xF8,
        0x98,
        0x11,
        0x69,
        0xD9,
        0x8E,
        0x94,
        0x9B,
        0x1E,
        0x87,
        0xE9,
        0xCE,
        0x55,
        0x28,
        0xDF,
        0x8C,
        0xA1,
        0x89,
        0x0D,
        0xBF,
        0xE6,
        0x42,
        0x68,
        0x41,
        0x99,
        0x2D,
        0x0F,
        0xB0,
        0x54,
        0xBB,
        0x16,
    ],
    dtype=np.uint8,
)


def _env_flag(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _env_limit(name, total, default):
    value = int(os.environ.get(name, str(default)))
    if value < 0:
        return total
    return min(value, total)


def _preprocess_mode():
    return os.environ.get("CHES25_PREPROCESS", "lucianoxu").strip().lower()


def _fit_trace_count(total):
    """Return how many profiling traces to use for CHES25 scaling stats.

    The official CHES2025 TensorFlow start kit fits StandardScaler on
    Profiling_traces[0:45000] and then applies that transform to attack traces.
    Keep 45000 as the default; use CHES25_STANDARDIZE_FIT_TRACES=-1 for all.
    """
    value = int(os.environ.get("CHES25_STANDARDIZE_FIT_TRACES", "45000"))
    if value < 0:
        return total
    return min(value, total)


def _stats_cache_path(data_path, width, fit_count):
    return os.environ.get(
        "CHES25_STANDARDIZE_STATS", "%s.estranet_standard_scaler_len%d_fit%d.npz" % (data_path, width, fit_count)
    )


def _compute_or_load_standard_scaler(corpus, data_path, width):
    profiling_traces = corpus["Profiling_traces"]["traces"]
    fit_count = _fit_trace_count(profiling_traces.shape[0])
    cache_path = _stats_cache_path(data_path, width, fit_count)

    if os.path.exists(cache_path):
        stats = np.load(cache_path)
        mean = stats["mean"].astype(np.float32)
        scale = stats["scale"].astype(np.float32)
        if mean.shape[0] == width and scale.shape[0] == width:
            print("[CHES25] Loaded StandardScaler stats from %s" % cache_path)
            return mean, scale
        print("[CHES25] Ignoring stale StandardScaler stats at %s" % cache_path)

    print("[CHES25] Fitting StandardScaler stats on %d profiling traces" % fit_count)
    batch_size = int(os.environ.get("CHES25_STANDARDIZE_CHUNK", "4096"))
    sums = np.zeros(width, dtype=np.float64)
    sq_sums = np.zeros(width, dtype=np.float64)
    seen = 0
    for start in range(0, fit_count, batch_size):
        stop = min(start + batch_size, fit_count)
        x = profiling_traces[start:stop, :width].astype(np.float32)
        sums += np.sum(x, axis=0, dtype=np.float64)
        sq_sums += np.sum(x * x, axis=0, dtype=np.float64)
        seen += x.shape[0]

    mean = sums / max(seen, 1)
    variance = (sq_sums / max(seen, 1)) - (mean * mean)
    variance = np.maximum(variance, 0.0)
    scale = np.sqrt(variance)
    scale[scale == 0.0] = 1.0
    mean = mean.astype(np.float32)
    scale = scale.astype(np.float32)

    tmp_path = "%s.%d.tmp.npz" % (cache_path, os.getpid())
    np.savez(tmp_path, mean=mean, scale=scale, fit_traces=np.array([seen]), width=np.array([width]))
    os.replace(tmp_path, cache_path)
    print("[CHES25] Wrote StandardScaler stats to %s" % cache_path)
    return mean, scale


def _apply_standard_scaler(traces, mean, scale):
    traces = traces.astype(np.float32, copy=False)
    traces -= mean[None, :]
    traces /= scale[None, :]
    return traces


def _lucianoxu_cache_path(data_path, width, profile_count, attack_count):
    denoise = "dn1" if _env_flag("CHES25_LX_DENOISE", True) else "dn0"
    return os.environ.get(
        "CHES25_LX_STATS",
        "%s.estranet_lucianoxu_%s_len%d_prof%d_att%d.npz" % (data_path, denoise, width, profile_count, attack_count),
    )


def _notch_mask(width):
    fs = float(os.environ.get("CHES25_LX_FS", "1000"))
    freqs_text = os.environ.get("CHES25_LX_NOTCH_FREQS", "20,25,30,40,60,80")
    freqs = [float(x) for x in freqs_text.replace(";", ",").split(",") if x.strip()]
    q = float(os.environ.get("CHES25_LX_NOTCH_Q", "100"))
    taper_ratio = float(os.environ.get("CHES25_LX_NOTCH_TAPER", "0.5"))
    atten_db = float(os.environ.get("CHES25_LX_NOTCH_ATTEN_DB", "20"))

    n_r = width // 2 + 1
    f_bins = np.arange(n_r, dtype=np.float32) * (fs / float(width))
    mask = np.ones(n_r, dtype=np.float32)
    att = np.float32(10.0 ** (-atten_db / 20.0))

    for f0 in freqs:
        if not (0.0 < f0 < fs * 0.5):
            continue
        half_bw = f0 / (2.0 * q)
        if half_bw <= 0:
            continue
        taper = max(taper_ratio * half_bw, 0.0)
        core = max(half_bw - taper, 0.0)
        delta = np.abs(f_bins - f0)
        if taper > 0:
            t = np.clip((delta - core) / taper, 0.0, 1.0)
            ramp = 0.5 - 0.5 * np.cos(np.pi * t)
            gain = att + (1.0 - att) * ramp
        else:
            gain = np.where(delta <= half_bw, att, 1.0)
        mask *= gain.astype(np.float32)
    return mask


def _denoise_trace_batch(traces, mask):
    if mask is None:
        return traces.astype(np.float32, copy=False)
    x = traces.astype(np.float32, copy=False)
    spectrum = np.fft.rfft(x, axis=1)
    spectrum *= mask[None, :]
    y = np.fft.irfft(spectrum, n=x.shape[1], axis=1).astype(np.float32)
    mix = float(os.environ.get("CHES25_LX_NOTCH_MIX", "1.0"))
    if mix != 1.0:
        y = mix * y + (1.0 - mix) * x
    return y.astype(np.float32, copy=False)


def _accumulate_stats(dataset, width, count, mask, key_bytes=None):
    batch_size = int(os.environ.get("CHES25_LX_CHUNK", "2048"))
    sums = np.zeros(width, dtype=np.float64)
    sq_sums = np.zeros(width, dtype=np.float64)
    seen = 0

    key127_sums = np.zeros(width, dtype=np.float64)
    key127_sq_sums = np.zeros(width, dtype=np.float64)
    key127_seen = 0

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        x = _denoise_trace_batch(dataset[start:stop, :width], mask)
        sums += np.sum(x, axis=0, dtype=np.float64)
        sq_sums += np.sum(x * x, axis=0, dtype=np.float64)
        seen += x.shape[0]

        if key_bytes is not None:
            sel = key_bytes[start:stop] == 127
            if np.any(sel):
                x_key = x[sel]
                key127_sums += np.sum(x_key, axis=0, dtype=np.float64)
                key127_sq_sums += np.sum(x_key * x_key, axis=0, dtype=np.float64)
                key127_seen += x_key.shape[0]

    def finalize(sum_v, sq_sum_v, n):
        mean = sum_v / max(n, 1)
        var = (sq_sum_v / max(n, 1)) - (mean * mean)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        std[std == 0.0] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    mean, std = finalize(sums, sq_sums, seen)
    if key_bytes is None:
        return mean, std, seen, None, None, 0

    key_mean, key_std = finalize(key127_sums, key127_sq_sums, key127_seen)
    return mean, std, seen, key_mean, key_std, key127_seen


def _compute_or_load_lucianoxu_stats(corpus, data_path, width):
    profiling = corpus["Profiling_traces"]["traces"]
    attack = corpus["Attack_traces"]["traces"]
    profile_count = _env_limit("CHES25_LX_PROFILE_TRACES", profiling.shape[0], 450000)
    attack_count = _env_limit("CHES25_LX_ATTACK_TRACES", attack.shape[0], -1)
    cache_path = _lucianoxu_cache_path(data_path, width, profile_count, attack_count)

    if os.path.exists(cache_path):
        stats = np.load(cache_path)
        if stats["mean_profiling"].shape[0] == width:
            print("[CHES25] Loaded LucianoXu drift stats from %s" % cache_path)
            return {k: stats[k].astype(np.float32) for k in stats.files}
        print("[CHES25] Ignoring stale LucianoXu drift stats at %s" % cache_path)

    print("[CHES25] Computing LucianoXu drift stats: profile=%d attack=%d" % (profile_count, attack_count))
    mask = _notch_mask(width) if _env_flag("CHES25_LX_DENOISE", True) else None
    metadata = corpus["Profiling_traces"]["metadata"]
    if metadata.dtype.names and "key" in metadata.dtype.names:
        key_bytes = metadata[:profile_count]["key"][:, TARGET_BYTE]
    else:
        key_bytes = np.array([metadata[i]["key"][TARGET_BYTE] for i in range(profile_count)])

    mean_prof, std_prof, seen_prof, mean_key127, std_key127, key127_seen = _accumulate_stats(
        profiling, width, profile_count, mask, key_bytes=key_bytes
    )
    mean_attack, std_attack, seen_attack, _, _, _ = _accumulate_stats(attack, width, attack_count, mask, key_bytes=None)

    stats = {
        "mean_profiling": mean_prof,
        "std_profiling": std_prof,
        "mean_attack": mean_attack,
        "std_attack": std_attack,
        "mean_profiling_raw_key_127": mean_key127,
        "std_profiling_raw_key_127": std_key127,
        "profile_count": np.array([seen_prof], dtype=np.int64),
        "attack_count": np.array([seen_attack], dtype=np.int64),
        "key127_count": np.array([key127_seen], dtype=np.int64),
    }
    tmp_path = "%s.%d.tmp.npz" % (cache_path, os.getpid())
    np.savez(tmp_path, **stats)
    os.replace(tmp_path, cache_path)
    print("[CHES25] Wrote LucianoXu drift stats to %s (key127=%d)" % (cache_path, key127_seen))
    return stats


def _apply_lucianoxu_preprocess(traces, split, stats, width):
    mask = _notch_mask(width) if _env_flag("CHES25_LX_DENOISE", True) else None
    batch_size = int(os.environ.get("CHES25_LX_CHUNK", "2048"))
    out = np.empty(traces.shape, dtype=np.float32)
    mean_prof = stats["mean_profiling"].astype(np.float32)
    std_prof = stats["std_profiling"].astype(np.float32)
    mean_attack = stats["mean_attack"].astype(np.float32)
    mean_key127 = stats["mean_profiling_raw_key_127"].astype(np.float32)

    for start in range(0, traces.shape[0], batch_size):
        stop = min(start + batch_size, traces.shape[0])
        x = _denoise_trace_batch(traces[start:stop, :width], mask)
        if split == "test":
            x = (x - mean_attack[None, :]) + mean_key127[None, :]
        x = (x - mean_prof[None, :]) / std_prof[None, :]
        out[start:stop] = x
    return out


class Dataset:
    def __init__(self, data_path, split, input_length, data_desync=0):
        self.data_path = data_path
        self.split = split
        self.input_length = input_length
        self.data_desync = data_desync

        corpus = h5py.File(data_path, "r")
        if split == "train":
            split_key = "Profiling_traces"
        elif split == "test":
            split_key = "Attack_traces"

        # Print available keys for debugging
        print("[CHES25] Available keys in '%s': %s" % (split_key, list(corpus[split_key].keys())))

        trace_width = self.input_length + self.data_desync
        self.traces = corpus[split_key]["traces"][:, :trace_width]
        self.num_samples = self.traces.shape[0]
        print("[CHES25] Loaded %d traces of shape %s" % (self.num_samples, self.traces.shape))

        mode = _preprocess_mode()
        if mode == "lucianoxu":
            stats = _compute_or_load_lucianoxu_stats(corpus, data_path, trace_width)
            self.traces = _apply_lucianoxu_preprocess(self.traces, split, stats, trace_width)
            print("[CHES25] Applied LucianoXu drift-fix preprocessing to %s traces" % split)
        elif mode == "official":
            mean, scale = _compute_or_load_standard_scaler(corpus, data_path, trace_width)
            self.traces = _apply_standard_scaler(self.traces, mean, scale)
            print("[CHES25] Applied official profiling-fitted StandardScaler to %s traces" % split)
        elif mode == "none" or not _env_flag("CHES25_STANDARDIZE", True):
            print("[CHES25] CHES25 preprocessing disabled")
        else:
            raise ValueError("Unsupported CHES25_PREPROCESS=%s" % mode)

        # Extract plaintexts and keys at TARGET_BYTE for evaluation and label computation
        metadata = corpus[split_key]["metadata"]
        self.plaintexts = self._extract_plaintexts(corpus, split_key, metadata)
        self.keys = self._extract_keys(corpus, split_key, metadata)

        # Load labels: use the official metadata labels when present, otherwise
        # compute the byte-0 AES S-box output from plaintext and key.
        if "labels" in corpus[split_key]:
            print("[CHES25] Using precomputed labels from dataset")
            self.labels = np.reshape(corpus[split_key]["labels"][()], [-1, 1])
            self.labels = self.labels.astype(np.int64)
        elif metadata.dtype.names and "labels" in metadata.dtype.names:
            print("[CHES25] Using labels from structured metadata")
            self.labels = np.reshape(metadata[:]["labels"], [-1, 1])
            self.labels = self.labels.astype(np.int64)
        else:
            print(
                "[CHES25] No 'labels' field found — computing from plaintext[%d] XOR key[%d] via AES S-box"
                % (TARGET_BYTE, TARGET_BYTE)
            )
            self.labels = AES_SBOX[np.bitwise_xor(self.plaintexts, self.keys)]
            self.labels = np.reshape(self.labels.astype(np.int64), [-1, 1])

        # Masks may not be present in the CHES CTF 2025 dataset
        try:
            self.masks = self._extract_masks(metadata)
        except (KeyError, IndexError, Exception):
            self.masks = None

        # Split into chunks to avoid large array allocations
        max_split_size = 2000000000 // self.input_length
        split_idx = list(range(max_split_size, self.num_samples, max_split_size))
        self.traces = np.split(self.traces, split_idx, axis=0)
        self.labels = np.split(self.labels, split_idx, axis=0)

    def _extract_plaintexts(self, corpus, split_key, metadata):
        """Extract plaintext byte at TARGET_BYTE. Handles both structured
        metadata arrays and separate 'plaintext' datasets."""
        # Try separate dataset first (e.g., corpus['Profiling_traces']['plaintext'])
        if "plaintext" in corpus[split_key]:
            pt = corpus[split_key]["plaintext"][()]
            if pt.ndim == 2:
                return pt[:, TARGET_BYTE]
            else:
                return pt
        if metadata.dtype.names and "plaintext" in metadata.dtype.names:
            return metadata[:]["plaintext"][:, TARGET_BYTE]
        # Fall back to structured metadata
        plaintexts = []
        for i in range(len(metadata)):
            plaintexts.append(metadata[i]["plaintext"][TARGET_BYTE])
        return np.array(plaintexts)

    def _extract_keys(self, corpus, split_key, metadata):
        """Extract key byte at TARGET_BYTE. Handles both structured
        metadata arrays and separate 'key' datasets."""
        # Try separate dataset first
        if "key" in corpus[split_key]:
            k = corpus[split_key]["key"][()]
            if k.ndim == 2:
                return k[:, TARGET_BYTE]
            else:
                return k
        if metadata.dtype.names and "key" in metadata.dtype.names:
            return metadata[:]["key"][:, TARGET_BYTE]
        # Fall back to structured metadata
        keys = []
        for i in range(len(metadata)):
            keys.append(metadata[i]["key"][TARGET_BYTE])
        return np.array(keys)

    def _extract_masks(self, metadata):
        """Extract masks from metadata if available."""
        masks = []
        for i in range(len(metadata)):
            masks.append(np.array(metadata[i]["masks"]))
        masks = np.stack(masks, axis=0)
        return masks

    def GetTFRecords(self, batch_size, training=False):
        dataset = tf.data.Dataset.from_tensor_slices((self.traces[0], self.labels[0]))
        for traces, labels in zip(self.traces[1:], self.labels[1:]):
            temp_dataset = tf.data.Dataset.from_tensor_slices((traces, labels))
            dataset.concatenate(temp_dataset)

        def shift(x, max_desync):
            ds = tf.random.uniform([1], 0, max_desync + 1, tf.dtypes.int32)
            ds = tf.concat([[0], ds], 0)
            x = tf.slice(x, ds, [-1, self.input_length])
            return x

        if training:
            if self.input_length < self.traces[0].shape[1]:
                return (
                    dataset.repeat()
                    .shuffle(self.num_samples)
                    .batch(batch_size // 2)
                    .map(lambda x, y: (shift(x, self.data_desync), y))
                    .unbatch()
                    .batch(batch_size, drop_remainder=True)
                    .map(lambda x, y: (tf.cast(x, tf.float32), y))
                    .prefetch(10)
                )
            else:
                return (
                    dataset.repeat()
                    .shuffle(self.num_samples)
                    .batch(batch_size, drop_remainder=True)
                    .map(lambda x, y: (tf.cast(x, tf.float32), y))
                    .prefetch(10)
                )

        else:
            if self.input_length < self.traces[0].shape[1]:
                return (
                    dataset.batch(batch_size, drop_remainder=True)
                    .map(lambda x, y: (shift(x, 0), y))
                    .map(lambda x, y: (tf.cast(x, tf.float32), y))
                    .prefetch(10)
                )
            else:
                return (
                    dataset.batch(batch_size, drop_remainder=True)
                    .map(lambda x, y: (tf.cast(x, tf.float32), y))
                    .prefetch(10)
                )

    def GetDataset(self):
        return self.traces, self.labels


if __name__ == "__main__":
    """Inspect the CHES CTF 2025 dataset structure and verify loading."""
    data_path = sys.argv[1]

    print("=" * 60)
    print("CHES CTF 2025 Dataset Inspector")
    print("=" * 60)
    print("File: %s" % data_path)

    f = h5py.File(data_path, "r")
    print("\nHDF5 structure:")

    def print_item(name, obj):
        if isinstance(obj, h5py.Dataset):
            print("  %-50s shape=%-20s dtype=%s" % (name, str(obj.shape), obj.dtype))
        else:
            print("  %s/" % name)

    f.visititems(print_item)

    print("\n" + "-" * 60)
    if len(sys.argv) >= 3:
        batch_size = int(sys.argv[2])
    else:
        batch_size = 16
    split = sys.argv[3] if len(sys.argv) >= 4 else "train"

    print("Loading split='%s' with input_length=100 ..." % split)
    dataset = Dataset(data_path, split, 100)

    print("\nDataset summary:")
    print("  traces    : %d chunks, first chunk shape=%s" % (len(dataset.traces), dataset.traces[0].shape))
    print("  labels    : %d chunks, first chunk shape=%s" % (len(dataset.labels), dataset.labels[0].shape))
    print("  plaintext : shape=%s" % str(dataset.plaintexts.shape))
    print("  keys      : shape=%s" % str(dataset.keys.shape))
    print("  traces dt : %s" % dataset.traces[0].dtype)
    print("  labels dt : %s" % dataset.labels[0].dtype)
    print("  target    : byte %d" % TARGET_BYTE)
    print("  masks     : %s" % ("present" if dataset.masks is not None else "not present"))

    print("\nFirst 5 samples:")
    for i in range(min(5, dataset.num_samples)):
        pt = dataset.plaintexts[i]
        k = dataset.keys[i]
        lbl = dataset.labels[0][i, 0] if i < len(dataset.labels[0]) else "?"
        expected = AES_SBOX[np.bitwise_xor(pt, k)]
        match = "OK" if int(lbl) == int(expected) else "MISMATCH"
        print("  [%d] pt=%3d key=%3d sbox[pt^key]=%3d label=%s (%s)" % (i, pt, k, expected, lbl, match))

    f.close()
