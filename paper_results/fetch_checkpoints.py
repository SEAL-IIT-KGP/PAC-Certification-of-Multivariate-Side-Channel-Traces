#!/usr/bin/env python3
"""Download the released checkpoints and unpack them into `checkpoints/`.

The archive `bi_certification_checkpoints.tar.gz` (308 MB) is a release file of this
repository. It holds the EstraNet Transformer checkpoints (`checkpoints/estranet/`) and the
two Keras models of ASCAD random key (`checkpoints/keras_ascad_random_key/`). The archive and
every unpacked file are verified against their SHA-256 checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

URL = (
    "https://github.com/SEAL-IIT-KGP/PAC-Certification-of-Multivariate-Side-Channel-Traces"
    "/releases/download/v1.0/bi_certification_checkpoints.tar.gz"
)
SHA256 = "8af739b7295a8ee7d92fef62976e9f8c3d5efac4f10e14fd796db8eea11e89a3"
FILES = 162


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpacked_files(repo: Path) -> int:
    folders = [repo / "checkpoints" / name for name in ("estranet", "keras_ascad_random_key")]
    return sum(1 for folder in folders for path in folder.rglob("*") if path.is_file())


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, help="An already downloaded archive to unpack.")
    parser.add_argument("--url", default=URL)
    parser.add_argument("--keep-archive", action="store_true", help="Keep the downloaded archive in checkpoints/.")
    args = parser.parse_args()

    if unpacked_files(repo) == FILES:
        print(f"{repo / 'checkpoints'}: all {FILES} files already unpacked")
        return

    archive = args.archive
    if archive is None:
        archive = repo / "checkpoints" / "bi_certification_checkpoints.tar.gz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial = archive.with_name(archive.name + ".part")
        print(f"downloading {args.url}")
        with urllib.request.urlopen(args.url) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
        partial.replace(archive)
    digest = sha256(archive)
    if digest != SHA256:
        raise SystemExit(f"{archive}: SHA-256 {digest} differs from the expected {SHA256}")
    print(f"{archive.name}: checksum verified")

    with tarfile.open(archive) as bundle:
        listing = bundle.extractfile("SHA256SUMS").read().decode()
        expected = dict(reversed(line.split(maxsplit=1)) for line in listing.splitlines() if line.strip())
        members = [member for member in bundle.getmembers() if member.name.startswith("checkpoints/")]
        options = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        bundle.extractall(repo, members=members, **options)
    changed = [name for name, digest in expected.items() if sha256(repo / name) != digest]
    if changed:
        raise SystemExit(f"{len(changed)} unpacked files differ from SHA256SUMS, for example {changed[0]}")
    print(f"unpacked and verified {len(expected)} files in {repo / 'checkpoints'}")

    if args.archive is None and not args.keep_archive:
        archive.unlink()


if __name__ == "__main__":
    main()
