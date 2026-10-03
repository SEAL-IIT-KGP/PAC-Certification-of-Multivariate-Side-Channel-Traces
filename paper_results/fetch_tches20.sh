#!/usr/bin/env bash
# Fetch the public TCHES20 benchmark (datasets, loaders, and the 480 pretrained
# models) and unpack it into the layout expected by the evaluators.
#
#   paper_results/fetch_tches20.sh [destination]
#
# Default destination: datasets/raw/tches20/TCHES20V3_CNN_SCA (about 2.3 GB unpacked).
set -euo pipefail

DEST="${1:-datasets/raw/tches20/TCHES20V3_CNN_SCA}"
URL="https://github.com/KULeuven-COSIC/TCHES20V3_CNN_SCA.git"
COMMIT="47e1b730691547224d48375bb0a4f1b5fd6d8b8c"

if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$(dirname "$DEST")"
  git clone "$URL" "$DEST"
fi
git -C "$DEST" checkout --quiet "$COMMIT"

unpack() {  # unpack <directory> <archive.zip> <marker path that exists once unpacked>
  local dir="$1" archive="$2" marker="$3"
  if [ -e "$dir/$marker" ]; then
    echo "already unpacked: $dir/$marker"
    return
  fi
  echo "unpacking $dir/$archive"
  if [ -f "$dir/${archive%.zip}.z01" ]; then
    # Split archive: join the parts first.
    (cd "$dir" && zip -q -s 0 "$archive" --out "joined_$archive" && unzip -q -o "joined_$archive" && rm -f "joined_$archive")
  else
    (cd "$dir" && unzip -q -o "$archive")
  fi
}

unpack "$DEST/datasets" ASCAD_dataset.zip ASCAD_dataset/ASCAD.h5
unpack "$DEST/datasets" AES_HD_dataset.zip AES_HD_dataset
unpack "$DEST/datasets" AES_RD_dataset.zip AES_RD_dataset
unpack "$DEST/datasets" DPAv4_dataset.zip DPAv4_dataset
unpack "$DEST/models" pretrained_models.zip pretrained_models

MODELS=$(find "$DEST/models/pretrained_models" -name '*.hdf5' | wc -l | tr -d ' ')
echo "TCHES20 benchmark ready in $DEST ($MODELS pretrained models)"
if [ "$MODELS" -ne 480 ]; then
  echo "expected 480 pretrained models" >&2
  exit 1
fi
