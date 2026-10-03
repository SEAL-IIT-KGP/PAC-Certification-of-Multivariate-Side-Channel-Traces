#!/usr/bin/env bash
# Fetch the EstraNet sources (Transformer model and ASCAD loader) and add the
# CHES-CTF-2025 loader used in the paper.
#
#   paper_results/fetch_estranet.sh [destination]
#
# Default destination: external/EstraNet
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
DEST="${1:-$REPO/external/EstraNet}"
URL="https://github.com/suvadeep-iitb/EstraNet.git"
COMMIT="7f969ce326c56dfbef8499136b6d9fe651033d00"

if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$(dirname "$DEST")"
  git clone "$URL" "$DEST"
fi
git -C "$DEST" checkout --quiet "$COMMIT"

# Files added on top of the upstream sources (see bi_suite_8datasets/estranet_support/README.md).
cp "$REPO/bi_suite_8datasets/estranet_support/data_utils_ches25.py" "$DEST/"
cp "$REPO/bi_suite_8datasets/estranet_support/evaluation_utils_ches25.py" "$DEST/"
echo "EstraNet sources ready in $DEST"
