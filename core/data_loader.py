"""
Unified Data Loader for Side-Channel Datasets

Supports:
- ASCAD (fixed-key and variable-key variants)
- AES-HD (Hamming Distance leakage model)

Provides consistent interfaces for:
- Loading raw traces and labels
- Creating train/validation/holdout splits
- Generating different representations (raw, POI, PCA)

Dataset Setup:
    Run `python -m datasets.setup_datasets --all` to download datasets first.
    See datasets/setup_datasets.py for download instructions.
"""

import numpy as np
from pathlib import Path

# Try to import h5py (only needed for ASCAD)
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
from typing import Tuple, Optional, Dict, List, Union, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings


# AES S-box and inverse
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
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]

AES_SBOX_INV = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
]


def hamming_weight(x: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    """Compute Hamming weight (number of 1 bits)."""
    if isinstance(x, np.ndarray):
        return np.array([bin(int(v)).count('1') for v in x])
    return bin(x).count('1')


# =============================================================================
# Dataset Path Discovery
# =============================================================================

def get_default_data_dir() -> Path:
    """Get the default data directory."""
    return Path(__file__).resolve().parents[1] / 'datasets' / 'raw'


def find_dataset(dataset_name: str, search_paths: List[str] = None) -> Optional[Path]:
    """
    Find a dataset file by searching common locations.

    Args:
        dataset_name: 'ascad' or 'aes_hd'
        search_paths: Additional paths to search

    Returns:
        Path to dataset file if found, None otherwise
    """
    if search_paths is None:
        search_paths = []

    # Build list of paths to search
    paths_to_check = []

    # Default data directory
    data_dir = get_default_data_dir()

    if dataset_name.lower() == 'ascad':
        filenames = ['ASCAD.h5', 'ASCAD_data/ASCAD.h5', 'ATMega8515_raw_traces.h5']
        paths_to_check.extend([
            data_dir / 'ascad' / fn for fn in filenames
        ])
    elif dataset_name.lower() in ['aes_hd', 'aes-hd', 'aeshd']:
        # Check for CSV directory first (from GitHub), then npz files
        csv_dir = data_dir / 'aes_hd' / 'AES_HD_Dataset'
        if csv_dir.exists() and (csv_dir / 'traces_1.csv').exists():
            return csv_dir
        filenames = ['aes_hd.npz', 'aes_hd_ext.npz', 'AES_HD_ext/aes_hd_ext.npz']
        paths_to_check.extend([
            data_dir / 'aes_hd' / fn for fn in filenames
        ])

    # Add user-specified search paths
    for sp in search_paths:
        paths_to_check.append(Path(sp))

    # Check current directory and parent
    cwd = Path.cwd()
    for fn in filenames if 'filenames' in dir() else []:
        paths_to_check.append(cwd / fn)
        paths_to_check.append(cwd / 'data' / fn)

    # Search for the file
    for path in paths_to_check:
        if path.exists():
            return path

    return None


def get_dataset_path(dataset_name: str, provided_path: str = None) -> Path:
    """
    Get the path to a dataset, with helpful error messages.

    Args:
        dataset_name: 'ascad' or 'aes_hd'
        provided_path: User-provided path (optional)

    Returns:
        Path to dataset file

    Raises:
        FileNotFoundError with helpful instructions
    """
    if provided_path:
        path = Path(provided_path)
        if path.exists():
            return path

    # Try to find automatically
    path = find_dataset(dataset_name)
    if path:
        return path

    # Dataset not found - provide helpful error
    data_dir = get_default_data_dir()

    if dataset_name.lower() == 'ascad':
        expected_path = data_dir / 'ascad' / 'ASCAD.h5'
        download_cmd = "python -m datasets.setup_datasets --ascad"
        manual_url = "https://github.com/ANSSI-FR/ASCAD"
    else:
        expected_path = data_dir / 'aes_hd' / 'aes_hd.npz'
        download_cmd = "python -m datasets.setup_datasets --aes-hd"
        manual_url = "https://github.com/AISyLab/AES_HD_Ext"

    raise FileNotFoundError(f"""
Dataset '{dataset_name}' not found.

Expected location: {expected_path}

To download, run:
    {download_cmd}

Or download manually from:
    {manual_url}

Then place the file at the expected location.
""")


@dataclass
class DataSplit:
    """Container for train/validation/holdout data splits."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_holdout: np.ndarray
    y_holdout: np.ndarray
    metadata: Dict


@dataclass
class DatasetInfo:
    """Dataset metadata container."""
    name: str
    n_samples: int
    n_features: int
    n_classes: int
    target_byte: int
    description: str


# =============================================================================
# ASCAD Dataset Loader
# =============================================================================

class ASCADLoader:
    """
    Loader for ASCAD dataset (ANSSI).

    ASCAD is an electromagnetic side-channel dataset from an 8-bit ATMega8515
    running a masked AES-128 implementation.

    Variants:
    - Fixed key: All traces use the same encryption key
    - Variable key: Different keys used across traces

    Usage:
        # With explicit path
        loader = ASCADLoader('datasets/raw/ascad/ASCAD.h5')

        # Auto-find in default locations
        loader = ASCADLoader.auto_load()
    """

    # Standard ASCAD POI indices (around first S-box operation)
    DEFAULT_POI = list(range(45400, 45550))  # Example range, adjust as needed

    def __init__(self, filepath: str = None, target_byte: int = 2):
        """
        Initialize ASCAD loader.

        Args:
            filepath: Path to ASCAD .h5 file (if None, will search default locations)
            target_byte: Target key byte index (default 2 for standard ASCAD)
        """
        if filepath is None:
            filepath = get_dataset_path('ascad')

        self.filepath = Path(filepath)
        self.target_byte = target_byte
        self._validate_file()

    @classmethod
    def auto_load(cls, target_byte: int = 2):
        """Create loader by automatically finding the dataset."""
        return cls(filepath=None, target_byte=target_byte)

    def _validate_file(self):
        """Validate that the file exists and is readable."""
        if not HAS_H5PY:
            raise ImportError("h5py is required to load ASCAD. Install with: pip install h5py")

        if not self.filepath.exists():
            raise FileNotFoundError(get_dataset_path.__doc__)

    def load(
        self,
        n_profiling: Optional[int] = None,
        n_attack: Optional[int] = None,
        window: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load ASCAD traces and labels.

        Args:
            n_profiling: Number of profiling traces (None = all)
            n_attack: Number of attack traces (None = all)
            window: (start, end) tuple for trace window selection

        Returns:
            X_profiling, y_profiling, X_attack, y_attack
        """
        with h5py.File(self.filepath, 'r') as f:
            # Load profiling set
            if n_profiling is None:
                X_prof = np.array(f['Profiling_traces/traces'])
                metadata_prof = np.array(f['Profiling_traces/metadata'])
            else:
                X_prof = np.array(f['Profiling_traces/traces'][:n_profiling])
                metadata_prof = np.array(f['Profiling_traces/metadata'][:n_profiling])

            # Load attack set
            if n_attack is None:
                X_attack = np.array(f['Attack_traces/traces'])
                metadata_attack = np.array(f['Attack_traces/metadata'])
            else:
                X_attack = np.array(f['Attack_traces/traces'][:n_attack])
                metadata_attack = np.array(f['Attack_traces/metadata'][:n_attack])

        # Apply window selection
        if window is not None:
            start, end = window
            X_prof = X_prof[:, start:end]
            X_attack = X_attack[:, start:end]

        # Extract labels (S-box output for target byte)
        y_prof = self._compute_labels(metadata_prof)
        y_attack = self._compute_labels(metadata_attack)

        return X_prof, y_prof, X_attack, y_attack

    def _compute_labels(self, metadata: np.ndarray) -> np.ndarray:
        """
        Compute target labels from metadata.

        For ASCAD, the label is typically: Sbox[plaintext[i] ^ key[i]]
        """
        # ASCAD metadata format: plaintext and key are stored in metadata
        # Adjust based on actual ASCAD format
        plaintexts = metadata['plaintext'][:, self.target_byte]
        keys = metadata['key'][:, self.target_byte]

        labels = np.array([AES_SBOX[p ^ k] for p, k in zip(plaintexts, keys)])
        return labels

    def get_info(self) -> DatasetInfo:
        """Get dataset information."""
        with h5py.File(self.filepath, 'r') as f:
            n_prof = f['Profiling_traces/traces'].shape[0]
            n_attack = f['Attack_traces/traces'].shape[0]
            n_features = f['Profiling_traces/traces'].shape[1]

        return DatasetInfo(
            name="ASCAD",
            n_samples=n_prof + n_attack,
            n_features=n_features,
            n_classes=256,
            target_byte=self.target_byte,
            description="ANSSI ASCAD electromagnetic traces"
        )


# =============================================================================
# AES-HD Dataset Loader
# =============================================================================

class AESHDLoader:
    """
    Loader for AES-HD dataset.

    AES-HD contains power traces from an unprotected AES implementation
    targeting the Hamming Distance leakage model.

    Supports both:
    - NPZ format (from AISyLab)
    - CSV format (from https://github.com/AESHD/AES_HD_Dataset)

    Usage:
        # With explicit path
        loader = AESHDLoader('datasets/raw/aes_hd/AES_HD_Dataset')

        # Auto-find in default locations
        loader = AESHDLoader.auto_load()
    """

    # Standard AES-HD POI indices
    DEFAULT_POI = [539, 540, 542, 543, 965, 966, 967, 968, 969, 1021]

    def __init__(self, filepath: str = None, target_byte: int = 11):
        """
        Initialize AES-HD loader.

        Args:
            filepath: Path to dataset (directory for CSV, file for NPZ)
            target_byte: Target byte index (default 11)
        """
        if filepath is None:
            filepath = get_dataset_path('aes_hd')

        self.filepath = Path(filepath)
        self.target_byte = target_byte
        self._is_csv_format = self.filepath.is_dir()
        self._validate_file()

    @classmethod
    def auto_load(cls, target_byte: int = 11):
        """Create loader by automatically finding the dataset."""
        return cls(filepath=None, target_byte=target_byte)

    def _validate_file(self):
        """Validate that the file/directory exists."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset not found at {self.filepath}")
        if self._is_csv_format:
            # Check for CSV files
            if not (self.filepath / 'traces_1.csv').exists():
                raise FileNotFoundError(f"traces_1.csv not found in {self.filepath}")

    def _load_csv_format(
        self,
        n_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load traces and labels from CSV files."""
        import pandas as pd

        # Find all trace files (traces_1.csv to traces_5.csv)
        trace_files = sorted(self.filepath.glob('traces_*.csv'))

        if not trace_files:
            raise FileNotFoundError(f"No trace CSV files found in {self.filepath}")

        print(f"Loading {len(trace_files)} trace files from {self.filepath}")

        # Load traces from all CSV files
        traces_list = []
        samples_loaded = 0

        for trace_file in trace_files:
            if n_samples is not None and samples_loaded >= n_samples:
                break

            print(f"  Loading {trace_file.name}...")
            df = pd.read_csv(trace_file, header=None)
            traces_list.append(df.values)
            samples_loaded += len(df)

        traces = np.vstack(traces_list).astype(np.float32)

        if n_samples is not None:
            traces = traces[:n_samples]

        # Load labels
        labels_file = self.filepath / 'labels.csv'
        if not labels_file.exists():
            # Try alternative names
            for alt_name in ['label.csv', 'Labels.csv', 'Label.csv']:
                alt_file = self.filepath / alt_name
                if alt_file.exists():
                    labels_file = alt_file
                    break

        if labels_file.exists():
            print(f"  Loading labels from {labels_file.name}...")
            labels_df = pd.read_csv(labels_file, header=None)
            labels = labels_df.values.flatten().astype(np.int32)
            if n_samples is not None:
                labels = labels[:n_samples]
        else:
            # If no labels file, generate dummy labels (user will need to provide)
            print(f"  Warning: No labels file found. Using placeholder labels.")
            labels = np.zeros(len(traces), dtype=np.int32)

        print(f"  Loaded {len(traces)} traces with {traces.shape[1]} features")

        return traces, labels

    def _load_npz_format(
        self,
        n_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load traces and metadata from NPZ file."""
        data = np.load(self.filepath)

        traces = np.array(data['traces'], dtype=np.float32)
        metadata = np.array(data['data'], dtype=np.uint8)

        if n_samples is not None:
            traces = traces[:n_samples]
            metadata = metadata[:n_samples]

        return traces, metadata

    def load(
        self,
        n_samples: Optional[int] = None,
        poi_indices: Optional[List[int]] = None,
        window: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load AES-HD traces and labels.

        Args:
            n_samples: Number of samples to load (None = all)
            poi_indices: List of POI indices to use (None = all or window)
            window: (start, end) tuple for trace window

        Returns:
            Traces, Labels
        """
        if self._is_csv_format:
            # CSV format: labels are pre-computed in separate file
            traces, labels = self._load_csv_format(n_samples)
        else:
            # NPZ format: compute labels from metadata
            traces, metadata = self._load_npz_format(n_samples)
            labels = self._compute_labels(metadata)

        # Apply POI or window selection
        if poi_indices is not None:
            traces = traces[:, poi_indices]
        elif window is not None:
            start, end = window
            traces = traces[:, start:end]

        return traces, labels

    def _compute_labels(self, metadata: np.ndarray) -> np.ndarray:
        """
        Compute target labels using Hamming Distance model.

        HD model: label = Sbox_inv[plaintext[i] ^ key[i]] ^ plaintext[j]
        """
        # Extract relevant bytes
        intermediate = metadata[:, 16:32]  # Intermediate values
        key = metadata[:, 32:]  # Key bytes

        input_1 = intermediate[:, self.target_byte].astype(np.uint16)
        input_2 = intermediate[:, 7].astype(np.uint16)  # Standard pairing
        key_byte = key[:, self.target_byte]

        # Compute HD target: Sbox_inv[input_1 ^ key] ^ input_2
        labels = np.array([
            AES_SBOX_INV[input_1[i] ^ key_byte[0]] ^ input_2[i]
            for i in range(len(input_1))
        ], dtype=np.uint16)

        return labels

    def get_info(self) -> DatasetInfo:
        """Get dataset information."""
        if self._is_csv_format:
            # For CSV, we need to check one file
            import pandas as pd
            sample_file = self.filepath / 'traces_1.csv'
            df = pd.read_csv(sample_file, header=None, nrows=1)
            n_features = df.shape[1]
            # Count total samples
            trace_files = list(self.filepath.glob('traces_*.csv'))
            n_samples = sum(
                sum(1 for _ in open(f)) for f in trace_files
            )
        else:
            data = np.load(self.filepath)
            traces = data['traces']
            n_samples = traces.shape[0]
            n_features = traces.shape[1]

        return DatasetInfo(
            name="AES-HD",
            n_samples=n_samples,
            n_features=n_features,
            n_classes=256,
            target_byte=self.target_byte,
            description="AES Hamming Distance power traces"
        )


# =============================================================================
# Unified Data Loading Interface
# =============================================================================

def _try_load_via_tches20(
    dataset_tag: str,
    data_dir: str = '',
    tches20_src_dir: str = '',
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Attempt to load a TCHES20 dataset using the TCHES20 repo's own dataLoaders.

    Returns (traces, labels) on success, or None if the loaders are not available.
    """
    import sys as _sys

    if tches20_src_dir:
        _sys.path.insert(0, tches20_src_dir)

    try:
        from dataLoaders import (
            load_ascad, load_aes_hd, load_aes_rd, load_dpav4,
        )
    except ImportError:
        return None

    import os as _os

    loader_map = {
        'ascad_desync_0':   lambda: load_ascad(
            _os.path.join(data_dir, 'ASCAD_dataset', 'ASCAD.h5')),
        'ascad_desync_50':  lambda: load_ascad(
            _os.path.join(data_dir, 'ASCAD_dataset', 'ASCAD_desync50.h5')),
        'ascad_desync_100': lambda: load_ascad(
            _os.path.join(data_dir, 'ASCAD_dataset', 'ASCAD_desync100.h5')),
        'aes_hd':           lambda: load_aes_hd(
            _os.path.join(data_dir, 'AES_HD_dataset/')),
        'aes_rd':           lambda: load_aes_rd(
            _os.path.join(data_dir, 'AES_RD_dataset/')),
        'dpav4':            lambda: load_dpav4(
            _os.path.join(data_dir, 'DPAv4_dataset/')),
    }

    if dataset_tag not in loader_map:
        return None

    try:
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
    except Exception as e:
        warnings.warn(f"TCHES20 loader failed for {dataset_tag}: {e}")
        return None


# TCHES20 dataset tags that require the external dataLoaders
TCHES20_TAGS = {
    'ascad_desync_0', 'ascad_desync_50', 'ascad_desync_100',
    'aes_rd', 'dpav4',
}


def load_dataset(
    dataset_name: str,
    filepath: str = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified dataset loading interface.

    Args:
        dataset_name: Dataset identifier. Supported values:
            - 'ascad': ASCAD fixed-key (via local ASCADLoader)
            - 'aes_hd': AES-HD (via local AESHDLoader)
            - 'ascad_desync_0', 'ascad_desync_50', 'ascad_desync_100',
              'aes_rd', 'dpav4': TCHES20 datasets (requires dataLoaders.py)
        filepath: Path to dataset file (optional - will auto-find if not provided)
        **kwargs: Dataset-specific arguments, including:
            - data_dir: Root directory containing TCHES20 dataset folders
            - tches20_src_dir: Path to TCHES20 repo src/ directory (for dataLoaders.py)
            - target_byte, n_profiling, n_attack, window, n_samples, poi_indices

    Returns:
        Traces, Labels

    Example:
        # Auto-find dataset
        traces, labels = load_dataset('aes_hd')

        # With explicit path
        traces, labels = load_dataset('ascad', 'datasets/raw/ascad/ASCAD.h5')

        # TCHES20 dataset
        traces, labels = load_dataset('ascad_desync_0',
            data_dir='datasets/raw/tches20/TCHES20V3_CNN_SCA/datasets',
            tches20_src_dir='datasets/raw/tches20/TCHES20V3_CNN_SCA/src')
    """
    dataset_name = dataset_name.lower()

    # Extract TCHES20-specific kwargs
    data_dir = kwargs.pop('data_dir', '')
    tches20_src_dir = kwargs.pop('tches20_src_dir', '')

    # --- TCHES20 datasets: try external dataLoaders first ---
    if dataset_name in TCHES20_TAGS:
        result = _try_load_via_tches20(dataset_name, data_dir, tches20_src_dir)
        if result is not None:
            return result
        raise ValueError(
            f"Dataset '{dataset_name}' requires TCHES20 dataLoaders but they could not be imported. "
            f"Ensure tches20_src_dir points to the TCHES20 repo src/ directory "
            f"(got: '{tches20_src_dir}') and that data_dir contains the dataset folders "
            f"(got: '{data_dir}')."
        )

    # --- aes_hd can also be loaded via TCHES20 or locally ---
    if dataset_name in ['aes_hd', 'aes-hd', 'aeshd']:
        # Try TCHES20 loader first if data_dir is provided
        if data_dir:
            result = _try_load_via_tches20('aes_hd', data_dir, tches20_src_dir)
            if result is not None:
                return result

        # Fallback to local loader
        if filepath is None or filepath == '':
            filepath = get_dataset_path(dataset_name)
        loader = AESHDLoader(filepath, target_byte=kwargs.get('target_byte', 11))
        traces, labels = loader.load(
            n_samples=kwargs.get('n_samples'),
            poi_indices=kwargs.get('poi_indices'),
            window=kwargs.get('window')
        )
        return traces, labels

    # --- ASCAD (local fixed-key) ---
    if dataset_name == 'ascad':
        if filepath is None or filepath == '':
            filepath = get_dataset_path(dataset_name)
        loader = ASCADLoader(filepath, target_byte=kwargs.get('target_byte', 2))
        X_prof, y_prof, X_attack, y_attack = loader.load(
            n_profiling=kwargs.get('n_profiling'),
            n_attack=kwargs.get('n_attack'),
            window=kwargs.get('window')
        )
        # Combine profiling and attack sets
        traces = np.vstack([X_prof, X_attack])
        labels = np.concatenate([y_prof, y_attack])
        return traces, labels

    raise ValueError(
        f"Unknown dataset: {dataset_name}. "
        f"Supported: 'ascad', 'aes_hd', 'ascad_desync_0', 'ascad_desync_50', "
        f"'ascad_desync_100', 'aes_rd', 'dpav4'"
    )


def create_data_split(
    traces: np.ndarray,
    labels: np.ndarray,
    train_size: float = 0.6,
    val_size: float = 0.2,
    holdout_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> DataSplit:
    """
    Create train/validation/holdout splits with proper protocol.

    CRITICAL: Holdout is used ONCE for certification, never for tuning.

    Args:
        traces: Full trace array
        labels: Full label array
        train_size: Fraction for training
        val_size: Fraction for validation (early stopping, hyperparams)
        holdout_size: Fraction for certification (used once!)
        random_state: Random seed for reproducibility
        stratify: Whether to stratify splits by label

    Returns:
        DataSplit object with all splits
    """
    assert abs(train_size + val_size + holdout_size - 1.0) < 1e-6, \
        "Split fractions must sum to 1.0"

    n_samples = len(labels)

    # First split: separate holdout
    stratify_arr = labels if stratify else None

    X_temp, X_holdout, y_temp, y_holdout = train_test_split(
        traces, labels,
        test_size=holdout_size,
        random_state=random_state,
        stratify=stratify_arr
    )

    # Second split: train vs validation
    val_ratio = val_size / (train_size + val_size)
    stratify_arr = y_temp if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=stratify_arr
    )

    metadata = {
        'n_total': n_samples,
        'n_train': len(y_train),
        'n_val': len(y_val),
        'n_holdout': len(y_holdout),
        'train_size': train_size,
        'val_size': val_size,
        'holdout_size': holdout_size,
        'random_state': random_state
    }

    return DataSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        metadata=metadata
    )


# =============================================================================
# Representation Transformations
# =============================================================================

def select_poi_by_snr(
    traces: np.ndarray,
    labels: np.ndarray,
    n_poi: int = 10
) -> Tuple[np.ndarray, List[int]]:
    """
    Select Points of Interest by Signal-to-Noise Ratio.

    SNR = Var(E[trace|label]) / E[Var(trace|label)]

    Args:
        traces: Trace array (n_samples, n_features)
        labels: Label array (n_samples,)
        n_poi: Number of POIs to select

    Returns:
        Selected trace subset, POI indices
    """
    n_samples, n_features = traces.shape
    n_classes = len(np.unique(labels))

    # Compute class-conditional means
    means = np.zeros((n_classes, n_features))
    vars = np.zeros((n_classes, n_features))
    counts = np.zeros(n_classes)

    for k in range(n_classes):
        idx = np.where(labels == k)[0]
        if len(idx) > 1:
            means[k] = np.mean(traces[idx], axis=0)
            vars[k] = np.var(traces[idx], axis=0)
            counts[k] = len(idx)

    # Compute SNR
    weights = counts / np.sum(counts)
    signal = np.var(means, axis=0)  # Variance of means
    noise = np.sum(weights.reshape(-1, 1) * vars, axis=0)  # Mean of variances
    snr = signal / (noise + 1e-10)

    # Select top POIs
    poi_indices = np.argsort(snr)[-n_poi:][::-1].tolist()

    return traces[:, poi_indices], poi_indices


def apply_pca(
    traces: np.ndarray,
    n_components: int,
    fit_traces: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, PCA]:
    """
    Apply PCA dimensionality reduction.

    Args:
        traces: Traces to transform
        n_components: Number of components
        fit_traces: Traces to fit PCA on (if different from traces)

    Returns:
        Transformed traces, fitted PCA object
    """
    pca = PCA(n_components=n_components)

    if fit_traces is not None:
        pca.fit(fit_traces)
        transformed = pca.transform(traces)
    else:
        transformed = pca.fit_transform(traces)

    return transformed, pca


def create_representations(
    data_split: DataSplit,
    methods: List[str],
    params: Optional[Dict] = None
) -> Dict[str, DataSplit]:
    """
    Create multiple representations of the data.

    Args:
        data_split: Original data split
        methods: List of representation methods
            - 'raw': Original traces
            - 'poi_N': Top N POIs by SNR
            - 'pca_N': PCA with N components
            - 'window_S_E': Window from index S to E
        params: Additional parameters

    Returns:
        Dictionary of DataSplit objects for each representation
    """
    params = params or {}
    representations = {}

    for method in methods:
        if method == 'raw':
            representations['raw'] = data_split

        elif method.startswith('poi_'):
            n_poi = int(method.split('_')[1])

            # Select POI based on training data
            X_train_poi, poi_idx = select_poi_by_snr(
                data_split.X_train, data_split.y_train, n_poi
            )

            representations[method] = DataSplit(
                X_train=X_train_poi,
                y_train=data_split.y_train,
                X_val=data_split.X_val[:, poi_idx],
                y_val=data_split.y_val,
                X_holdout=data_split.X_holdout[:, poi_idx],
                y_holdout=data_split.y_holdout,
                metadata={**data_split.metadata, 'poi_indices': poi_idx}
            )

        elif method.startswith('pca_'):
            n_components = int(method.split('_')[1])

            # Fit PCA on training data only
            X_train_pca, pca = apply_pca(
                data_split.X_train, n_components
            )

            representations[method] = DataSplit(
                X_train=X_train_pca,
                y_train=data_split.y_train,
                X_val=pca.transform(data_split.X_val),
                y_val=data_split.y_val,
                X_holdout=pca.transform(data_split.X_holdout),
                y_holdout=data_split.y_holdout,
                metadata={**data_split.metadata, 'pca_variance_ratio': pca.explained_variance_ratio_.tolist()}
            )

        elif method.startswith('window_'):
            parts = method.split('_')
            start, end = int(parts[1]), int(parts[2])

            representations[method] = DataSplit(
                X_train=data_split.X_train[:, start:end],
                y_train=data_split.y_train,
                X_val=data_split.X_val[:, start:end],
                y_val=data_split.y_val,
                X_holdout=data_split.X_holdout[:, start:end],
                y_holdout=data_split.y_holdout,
                metadata={**data_split.metadata, 'window': (start, end)}
            )

    return representations


# =============================================================================
# Synthetic Data Generation (for testing)
# =============================================================================

def generate_synthetic_data(
    n_samples: int = 50000,
    n_features: int = 100,
    n_classes: int = 256,
    snr: float = 1.0,
    n_leaky_points: int = 10,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic side-channel data for testing.

    Args:
        n_samples: Number of traces
        n_features: Number of time samples per trace
        n_classes: Number of classes (256 for AES byte)
        snr: Signal-to-noise ratio
        n_leaky_points: Number of leaky points in trace
        random_state: Random seed

    Returns:
        Traces, Labels
    """
    np.random.seed(random_state)

    # Generate random labels
    labels = np.random.randint(0, n_classes, n_samples)

    # Compute leakage model (Hamming weight)
    hw = hamming_weight(labels)

    # Generate traces with leakage at specific points
    noise_std = 1.0 / np.sqrt(snr)
    traces = np.random.randn(n_samples, n_features) * noise_std

    # Add leakage at random positions
    leaky_positions = np.random.choice(n_features, n_leaky_points, replace=False)
    for pos in leaky_positions:
        traces[:, pos] += hw

    return traces, labels


def check_datasets() -> Dict[str, Dict]:
    """
    Check which datasets are available.

    Returns:
        Dictionary with dataset availability info
    """
    results = {}

    for name in ['ascad', 'aes_hd']:
        try:
            path = find_dataset(name)
            if path:
                results[name] = {
                    'available': True,
                    'path': str(path),
                    'size_mb': path.stat().st_size / (1024 * 1024)
                }
            else:
                results[name] = {
                    'available': False,
                    'path': None,
                    'message': 'Not found. Run: python -m datasets.setup_datasets --' + name.replace('_', '-')
                }
        except Exception as e:
            results[name] = {
                'available': False,
                'path': None,
                'message': str(e)
            }

    return results


def print_dataset_status():
    """Print status of all datasets."""
    print("\nDataset Status")
    print("=" * 50)

    status = check_datasets()

    for name, info in status.items():
        if info['available']:
            print(f"  ✓ {name}: {info['path']}")
            print(f"      Size: {info['size_mb']:.1f} MB")
        else:
            print(f"  ✗ {name}: NOT FOUND")
            if 'message' in info:
                print(f"      {info['message']}")

    print()


if __name__ == "__main__":
    # Check dataset availability first
    print_dataset_status()

    # Example usage with synthetic data
    print("Data Loader Module - Example Usage")
    print("=" * 50)

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    traces, labels = generate_synthetic_data(
        n_samples=10000,
        n_features=50,
        n_classes=256,
        snr=0.5,
        n_leaky_points=5
    )
    print(f"  Traces shape: {traces.shape}")
    print(f"  Labels shape: {labels.shape}")

    # Create data split
    print("\nCreating train/val/holdout split...")
    split = create_data_split(
        traces, labels,
        train_size=0.6,
        val_size=0.2,
        holdout_size=0.2
    )
    print(f"  Training: {len(split.y_train)}")
    print(f"  Validation: {len(split.y_val)}")
    print(f"  Holdout: {len(split.y_holdout)}")

    # Create representations
    print("\nCreating representations...")
    reps = create_representations(
        split,
        methods=['raw', 'poi_5', 'pca_10', 'window_10_30']
    )
    for name, rep in reps.items():
        print(f"  {name}: shape = {rep.X_train.shape}")
