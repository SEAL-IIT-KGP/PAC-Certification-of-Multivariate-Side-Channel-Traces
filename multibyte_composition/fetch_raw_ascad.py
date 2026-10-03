#!/usr/bin/env python3
"""Fetch the raw ASCAD fixed-key traces (100,000 samples per trace) and verify the checksum.

The file `ATMega8515_raw_traces.h5` (6.0 GB, 60,000 traces) ships inside the ANSSI
ASCAD archive `ASCAD_data.zip` (4.4 GB; https://github.com/ANSSI-FR/ASCAD,
ATMEGA_AES_v1/ATM_AES_v1_fixed_key). The archive is downloaded, checked against the
published SHA-256, and only the raw-trace member is extracted to
`datasets/raw/ascad_raw/ATMega8515_raw_traces.h5`, which is checked in turn. Both
hashes are the ones listed in the ASCAD README. If you already hold the archive or
the raw file, pass `--zip PATH` or place the file at the output path: it is then only
verified. The download refuses a resource whose announced size differs from the
archive's, so a wrong link cannot fill the disk.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

ZIP_URL = "https://www.data.gouv.fr/api/1/datasets/r/e7ab6f9e-79bf-431f-a5ed-faf0ebe9b08e"
ZIP_SIZE = 4_435_199_469
ZIP_SHA256 = "a6884faf97133f9397aeb1af247dc71ab7616f3c181190f127ea4c474a0ad72c"
MEMBER_SUFFIX = "ASCAD_databases/ATMega8515_raw_traces.h5"
RAW_SIZE = 6_003_842_144
RAW_SHA256 = "51e722f6c63a590ce2c4633c9a9534e8e1b22a9cde8e4532e32c11ac089d4625"
CHUNK = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str, label: str) -> None:
    digest = sha256(path)
    if digest != expected:
        raise SystemExit(f"{path}: SHA-256 {digest} differs from the expected {expected} ({label})")
    print(f"{path}: checksum verified ({label})")


def download_zip(target: Path) -> None:
    partial = target.with_suffix(".part")
    with urllib.request.urlopen(ZIP_URL) as response:
        announced = response.headers.get("Content-Length")
        if announced is not None and abs(int(announced) - ZIP_SIZE) > ZIP_SIZE // 100:
            raise SystemExit(
                f"{ZIP_URL} announces {int(announced):,} bytes, but ASCAD_data.zip is {ZIP_SIZE:,} bytes; refusing to download"
            )
        print(f"downloading ASCAD_data.zip ({ZIP_SIZE / 1e9:.1f} GB) from {ZIP_URL}")
        written = 0
        with partial.open("wb") as handle:
            for chunk in iter(lambda: response.read(CHUNK), b""):
                written += len(chunk)
                if written > ZIP_SIZE + CHUNK:
                    handle.close()
                    partial.unlink(missing_ok=True)
                    raise SystemExit(f"{ZIP_URL} sent more than {ZIP_SIZE:,} bytes; the resource is not ASCAD_data.zip")
                handle.write(chunk)
    partial.replace(target)


def extract_raw(archive: Path, output: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist() if name.endswith(MEMBER_SUFFIX)]
        if len(members) != 1:
            raise SystemExit(f"{archive}: expected one member ending in {MEMBER_SUFFIX}, found {members}")
        print(f"extracting {members[0]} ({RAW_SIZE / 1e9:.1f} GB)")
        partial = output.with_suffix(".part")
        with bundle.open(members[0]) as source, partial.open("wb") as handle:
            shutil.copyfileobj(source, handle, length=CHUNK)
        partial.replace(output)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=repo / "datasets/raw/ascad_raw/ATMega8515_raw_traces.h5")
    parser.add_argument("--zip", type=Path, default=None, help="An already downloaded ASCAD_data.zip to extract from.")
    parser.add_argument("--keep-zip", action="store_true", help="Keep the downloaded archive next to the output.")
    args = parser.parse_args()
    if args.output.exists():
        verify(args.output, RAW_SHA256, "raw traces")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    archive = args.zip or args.output.parent / "ASCAD_data.zip"
    downloaded = False
    if not archive.exists():
        download_zip(archive)
        downloaded = True
    verify(archive, ZIP_SHA256, "ASCAD_data.zip")
    extract_raw(archive, args.output)
    verify(args.output, RAW_SHA256, "raw traces")
    if downloaded and not args.keep_zip:
        archive.unlink()
        print(f"removed {archive}")


if __name__ == "__main__":
    main()
