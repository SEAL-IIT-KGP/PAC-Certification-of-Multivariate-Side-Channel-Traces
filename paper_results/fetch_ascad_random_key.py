#!/usr/bin/env python3
"""Download the ASCAD random-key database (1,400-sample window) and verify its checksum.

The file is `ascad-variable.h5` of the ANSSI ASCAD release
(https://github.com/ANSSI-FR/ASCAD, ATMEGA_AES_v1/ATM_AES_v1_variable_key),
stored as `datasets/raw/ascad_random_key/ASCAD.h5` (about 420 MB).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

URL = "https://www.data.gouv.fr/api/1/datasets/r/b4ace767-c2a4-4db4-8e01-4527b5b91f00"
SHA256 = "d834da6ca5a288c4ba5add8e336845270a055d6eaf854dcf2f325a2eb6d7de06"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=repo / "datasets/raw/ascad_random_key/ASCAD.h5")
    args = parser.parse_args()

    if not args.output.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_suffix(".part")
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
        partial.replace(args.output)
    digest = sha256(args.output)
    if digest != SHA256:
        raise SystemExit(f"{args.output}: SHA-256 {digest} differs from the expected {SHA256}")
    print(f"{args.output}: checksum verified")


if __name__ == "__main__":
    main()
