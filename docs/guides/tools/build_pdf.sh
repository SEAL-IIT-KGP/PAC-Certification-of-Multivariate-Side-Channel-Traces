#!/usr/bin/env bash
# Builds one PDF per chapter into pdf/ (pandoc + XeLaTeX).  `build_pdf.sh 03` builds one chapter.
set -euo pipefail
G="$(cd "$(dirname "$0")/.." && pwd)"; cd "$G"; mkdir -p pdf
for md in chapters/${1:-}*.md; do
  base="$(basename "${md%.md}")"
  pandoc metadata.yaml "$md" -o "pdf/$base.pdf" \
    --pdf-engine=xelatex --lua-filter tools/include.lua --lua-filter tools/noalt.lua \
    --highlight-style tango \
    -V mainfont="Helvetica Neue" -V monofont="Menlo" -V monofontoptions="Scale=0.78" \
    -V geometry:margin=2cm -V fontsize=10.5pt -V colorlinks=true -V linkcolor=NavyBlue -V urlcolor=NavyBlue \
    -V documentclass=article -H tools/header.tex
  echo "built pdf/$base.pdf"
done
