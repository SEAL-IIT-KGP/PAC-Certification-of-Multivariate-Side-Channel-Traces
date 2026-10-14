#!/usr/bin/env bash
# Copies the figures the guides show from the repository's results/ into guide/figures/.
set -u
REPO="${GUIDE_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
G="$(cd "$(dirname "$0")/.." && pwd)"; mkdir -p "$G/images"
for f in paper_results/figure1_dimension_sweep.png paper_results/figure1_dimension_sweep_delta1e-6.png \
  paper_results/figure2_biloc_selected_centers.png; do
  if [ -f "$REPO/results/$f" ]; then cp "$REPO/results/$f" "$G/images/$(basename "$f")"; echo "copied $f"; else echo "missing $f"; fi
done
