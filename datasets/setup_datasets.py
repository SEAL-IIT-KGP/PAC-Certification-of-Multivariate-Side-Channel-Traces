#!/usr/bin/env python3
"""
Dataset Setup Script for BI Certification Experiments

Downloads and prepares:
- ASCAD (ANSSI): Electromagnetic traces, masked AES implementation
- AES-HD: Power traces, Hamming Distance leakage model

Usage:
    python -m datasets.setup_datasets --all      # Download all datasets
    python -m datasets.setup_datasets --ascad    # Download ASCAD only
    python -m datasets.setup_datasets --aes-hd   # Download AES-HD only
    python -m datasets.setup_datasets --verify   # Verify existing datasets
"""

import os
import sys
import argparse
import hashlib
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import Optional, Dict
import shutil

# Try to import h5py and numpy for verification
try:
    import h5py
    import numpy as np
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Warning: h5py/numpy not installed. Install with: pip install h5py numpy")


# =============================================================================
# Dataset Configuration
# =============================================================================

DATASETS = {
    'ascad_fixed': {
        'name': 'ASCAD (Fixed Key)',
        'description': 'ANSSI electromagnetic traces, fixed key variant',
        'url': 'https://static.data.gouv.fr/resources/ascad/20180530-163000/ASCAD_data.zip',
        'filename': 'ASCAD_data.zip',
        'extracted_name': 'ASCAD.h5',
        'size_mb': 4700,  # Approximate
        'md5': None,  # Add if known
        'n_profiling': 50000,
        'n_attack': 10000,
        'n_features': 700,
        'target_byte': 2,
    },
    'ascad_variable': {
        'name': 'ASCAD (Variable Key)',
        'description': 'ANSSI electromagnetic traces, variable key variant',
        'url': 'https://static.data.gouv.fr/resources/ascad/20180530-163000/ASCAD_data.zip',
        'filename': 'ASCAD_data.zip',
        'extracted_name': 'ASCAD.h5',
        'size_mb': 4700,
        'md5': None,
    },
    'aes_hd': {
        'name': 'AES-HD Extended',
        'description': 'Power traces with Hamming Distance leakage',
        'github_repo': 'https://github.com/AISyLab/AES_HD_Ext',
        'direct_url': 'https://github.com/AISyLab/AES_HD_Ext/raw/master/aes_hd_ext.npz',
        'filename': 'aes_hd_ext.npz',
        'size_mb': 500,  # Approximate
        'md5': None,
        'n_samples': 500000,
        'n_features': 1250,
        'poi_indices': [539, 540, 542, 543, 965, 966, 967, 968, 969, 1021],
    },
}

# Default data directory
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / 'raw'


# =============================================================================
# Download Utilities
# =============================================================================

def get_file_size_str(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def download_with_progress(url: str, dest_path: Path, desc: str = "Downloading"):
    """Download file with progress indicator."""
    print(f"\n{desc}")
    print(f"  URL: {url}")
    print(f"  Destination: {dest_path}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Get file size
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            print(f"  Size: {get_file_size_str(total_size)}")

            # Download with progress
            downloaded = 0
            block_size = 8192 * 16  # 128KB blocks

            with open(dest_path, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break

                    downloaded += len(buffer)
                    f.write(buffer)

                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        bar_len = 40
                        filled = int(bar_len * downloaded / total_size)
                        bar = '=' * filled + '-' * (bar_len - filled)
                        print(f"\r  Progress: [{bar}] {percent:.1f}%", end='', flush=True)

            print()  # New line after progress bar

        print(f"  ✓ Download complete: {dest_path}")
        return True

    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def compute_md5(filepath: Path) -> str:
    """Compute MD5 hash of file."""
    md5_hash = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


# =============================================================================
# ASCAD Dataset
# =============================================================================

def download_ascad(data_dir: Path, variant: str = 'fixed') -> bool:
    """
    Download ASCAD dataset.

    ASCAD is hosted on data.gouv.fr (French government data portal).
    The download is a large ZIP file containing the HDF5 dataset.

    Args:
        data_dir: Directory to store dataset
        variant: 'fixed' or 'variable' key

    Returns:
        True if successful
    """
    config = DATASETS['ascad_fixed'] if variant == 'fixed' else DATASETS['ascad_variable']
    ascad_dir = data_dir / 'ascad'
    ascad_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"ASCAD Dataset ({variant} key)")
    print(f"{'='*60}")
    print(f"Description: {config['description']}")
    print(f"Estimated size: ~{config['size_mb']} MB")

    # Check if already exists
    final_path = ascad_dir / 'ASCAD.h5'
    if final_path.exists():
        print(f"\n✓ Dataset already exists: {final_path}")
        return verify_ascad(final_path)

    # For ASCAD, we provide instructions since the file is very large
    print("\n" + "="*60)
    print("ASCAD DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("""
The ASCAD dataset is large (~4.7 GB) and hosted on the French government
data portal. Due to its size, automatic download may be unreliable.

Option 1: Manual Download (Recommended)
---------------------------------------
1. Visit: https://github.com/ANSSI-FR/ASCAD
2. Download the dataset from the provided Google Drive links
3. Extract and place ASCAD.h5 in: {ascad_dir}

Option 2: Direct Download (May be slow/unstable)
------------------------------------------------
URL: {url}

Option 3: Use wget/curl (from terminal)
---------------------------------------
cd {ascad_dir}
wget -O ASCAD_data.zip "{url}"
unzip ASCAD_data.zip

After downloading, the file should be at:
  {final_path}
""".format(ascad_dir=ascad_dir, url=config['url'], final_path=final_path))

    # Ask user if they want to attempt automatic download
    try:
        response = input("\nAttempt automatic download? (y/N): ").strip().lower()
        if response == 'y':
            zip_path = ascad_dir / config['filename']
            if download_with_progress(config['url'], zip_path, "Downloading ASCAD..."):
                print("Extracting...")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(ascad_dir)
                print(f"✓ Extracted to {ascad_dir}")

                # Find the .h5 file
                h5_files = list(ascad_dir.rglob("*.h5"))
                if h5_files:
                    # Move to expected location
                    shutil.move(str(h5_files[0]), str(final_path))
                    print(f"✓ Dataset ready: {final_path}")

                # Clean up zip
                zip_path.unlink()
                return True
    except KeyboardInterrupt:
        print("\nDownload cancelled.")
    except EOFError:
        print("\nNon-interactive mode - skipping automatic download.")

    return final_path.exists()


def verify_ascad(filepath: Path) -> bool:
    """Verify ASCAD dataset integrity."""
    if not HAS_DEPS:
        print("  Skipping verification (h5py not installed)")
        return True

    print(f"\nVerifying ASCAD dataset: {filepath}")

    try:
        with h5py.File(filepath, 'r') as f:
            # Check expected structure
            assert 'Profiling_traces' in f, "Missing Profiling_traces group"
            assert 'Attack_traces' in f, "Missing Attack_traces group"

            prof_traces = f['Profiling_traces/traces']
            attack_traces = f['Attack_traces/traces']

            print(f"  Profiling traces: {prof_traces.shape}")
            print(f"  Attack traces: {attack_traces.shape}")
            print(f"  ✓ ASCAD verification passed")
            return True

    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        return False


# =============================================================================
# AES-HD Dataset
# =============================================================================

def download_aes_hd(data_dir: Path) -> bool:
    """
    Download AES-HD Extended dataset.

    This dataset is available from GitHub and is smaller than ASCAD.

    Args:
        data_dir: Directory to store dataset

    Returns:
        True if successful
    """
    config = DATASETS['aes_hd']
    aes_hd_dir = data_dir / 'aes_hd'
    aes_hd_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"AES-HD Extended Dataset")
    print(f"{'='*60}")
    print(f"Description: {config['description']}")
    print(f"Estimated size: ~{config['size_mb']} MB")
    print(f"Source: {config['github_repo']}")

    final_path = aes_hd_dir / config['filename']

    # Check if already exists
    if final_path.exists():
        print(f"\n✓ Dataset already exists: {final_path}")
        return verify_aes_hd(final_path)

    # Try direct download from GitHub
    print("\nAttempting download from GitHub...")

    # GitHub LFS files may need special handling
    # First try direct URL
    if download_with_progress(config['direct_url'], final_path, "Downloading AES-HD..."):
        return verify_aes_hd(final_path)

    # If direct download fails, provide manual instructions
    print("\n" + "="*60)
    print("AES-HD DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print(f"""
If automatic download failed, please download manually:

Option 1: Clone the repository
------------------------------
git clone {config['github_repo']} {aes_hd_dir}

Option 2: Download directly
---------------------------
Visit: {config['github_repo']}
Download: aes_hd_ext.npz
Place in: {aes_hd_dir}

The file should be at:
  {final_path}
""")

    return final_path.exists()


def verify_aes_hd(filepath: Path) -> bool:
    """Verify AES-HD dataset integrity."""
    if not HAS_DEPS:
        print("  Skipping verification (numpy not installed)")
        return True

    print(f"\nVerifying AES-HD dataset: {filepath}")

    try:
        data = np.load(filepath)

        # Check expected keys
        assert 'traces' in data, "Missing 'traces' array"
        assert 'data' in data, "Missing 'data' (metadata) array"

        traces = data['traces']
        metadata = data['data']

        print(f"  Traces shape: {traces.shape}")
        print(f"  Metadata shape: {metadata.shape}")

        # Verify expected POI indices are accessible
        config = DATASETS['aes_hd']
        max_poi = max(config['poi_indices'])
        assert traces.shape[1] > max_poi, f"Traces too short for POI indices (need >{max_poi})"

        print(f"  ✓ AES-HD verification passed")
        return True

    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        return False


# =============================================================================
# Main Functions
# =============================================================================

def setup_all_datasets(data_dir: Path) -> Dict[str, bool]:
    """Download and verify all datasets."""
    results = {}

    print("\n" + "="*70)
    print("BI Certification Experiments - Dataset Setup")
    print("="*70)
    print(f"Data directory: {data_dir}")

    # ASCAD
    results['ascad'] = download_ascad(data_dir, variant='fixed')

    # AES-HD
    results['aes_hd'] = download_aes_hd(data_dir)

    # Summary
    print("\n" + "="*70)
    print("SETUP SUMMARY")
    print("="*70)
    for name, success in results.items():
        status = "✓ Ready" if success else "✗ Not available"
        print(f"  {name}: {status}")

    return results


def verify_datasets(data_dir: Path) -> Dict[str, bool]:
    """Verify all existing datasets."""
    results = {}

    print("\n" + "="*70)
    print("Dataset Verification")
    print("="*70)

    # ASCAD
    ascad_path = data_dir / 'ascad' / 'ASCAD.h5'
    if ascad_path.exists():
        results['ascad'] = verify_ascad(ascad_path)
    else:
        print(f"\nASCAD: Not found at {ascad_path}")
        results['ascad'] = False

    # AES-HD
    aes_hd_path = data_dir / 'aes_hd' / 'aes_hd_ext.npz'
    if aes_hd_path.exists():
        results['aes_hd'] = verify_aes_hd(aes_hd_path)
    else:
        print(f"\nAES-HD: Not found at {aes_hd_path}")
        results['aes_hd'] = False

    return results


def get_dataset_paths(data_dir: Path = None) -> Dict[str, Optional[Path]]:
    """
    Get paths to available datasets.

    Args:
        data_dir: Data directory (default: datasets/raw)

    Returns:
        Dictionary mapping dataset names to paths (None if not available)
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    paths = {}

    # ASCAD
    ascad_path = data_dir / 'ascad' / 'ASCAD.h5'
    paths['ascad'] = ascad_path if ascad_path.exists() else None

    # AES-HD
    aes_hd_path = data_dir / 'aes_hd' / 'aes_hd_ext.npz'
    paths['aes_hd'] = aes_hd_path if aes_hd_path.exists() else None

    return paths


def print_dataset_info():
    """Print information about all datasets."""
    print("\n" + "="*70)
    print("Dataset Information")
    print("="*70)

    for key, config in DATASETS.items():
        print(f"\n{config['name']}")
        print("-" * 40)
        print(f"  Description: {config['description']}")
        print(f"  Size: ~{config['size_mb']} MB")
        if 'n_samples' in config:
            print(f"  Samples: {config['n_samples']:,}")
        if 'n_profiling' in config:
            print(f"  Profiling: {config['n_profiling']:,}")
            print(f"  Attack: {config['n_attack']:,}")
        if 'n_features' in config:
            print(f"  Features: {config['n_features']}")


def main():
    parser = argparse.ArgumentParser(
        description='Setup datasets for BI Certification experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m datasets.setup_datasets --all      # Download all datasets
  python -m datasets.setup_datasets --ascad    # Download ASCAD only
  python -m datasets.setup_datasets --aes-hd   # Download AES-HD only
  python -m datasets.setup_datasets --verify   # Verify existing datasets
  python -m datasets.setup_datasets --info     # Show dataset information
        """
    )

    parser.add_argument('--all', action='store_true',
                       help='Download all datasets')
    parser.add_argument('--ascad', action='store_true',
                       help='Download ASCAD dataset')
    parser.add_argument('--aes-hd', action='store_true',
                       help='Download AES-HD dataset')
    parser.add_argument('--verify', action='store_true',
                       help='Verify existing datasets')
    parser.add_argument('--info', action='store_true',
                       help='Show dataset information')
    parser.add_argument('--data-dir', type=str, default=str(DEFAULT_DATA_DIR),
                       help=f'Data directory (default: {DEFAULT_DATA_DIR})')

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if args.info:
        print_dataset_info()
        return

    if args.verify:
        results = verify_datasets(data_dir)
        sys.exit(0 if all(results.values()) else 1)

    if args.all:
        results = setup_all_datasets(data_dir)
    elif args.ascad:
        results = {'ascad': download_ascad(data_dir)}
    elif args.aes_hd:
        results = {'aes_hd': download_aes_hd(data_dir)}
    else:
        parser.print_help()
        print("\n⚠ No action specified. Use --all, --ascad, --aes-hd, or --verify")
        return

    # Print paths for use in experiments
    print("\n" + "="*70)
    print("Dataset Paths (for experiments)")
    print("="*70)
    paths = get_dataset_paths(data_dir)
    for name, path in paths.items():
        if path:
            print(f"  {name}: {path}")
        else:
            print(f"  {name}: NOT AVAILABLE")


if __name__ == '__main__':
    main()
