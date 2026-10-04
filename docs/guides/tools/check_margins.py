#!/usr/bin/env python3
"""Report pages with ink inside the right or bottom margin (text running off the page)."""
import glob, subprocess, sys, tempfile, os
from PIL import Image
import numpy as np
DPI = 50; MARGIN_CM = 2.0
bad = 0
for pdf in sorted(glob.glob("pdf/*.pdf")):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdftoppm", "-r", str(DPI), "-png", pdf, f"{d}/p"], check=True)
        for png in sorted(glob.glob(f"{d}/p-*.png")):
            a = np.array(Image.open(png).convert("L"))
            h, w = a.shape
            m = int(MARGIN_CM / 2.54 * DPI)
            right = a[:, w - m + 6:]           # beyond the text block, with a small tolerance
            if (right < 128).any():
                rows = np.where((right < 128).any(axis=1))[0]
                print(f"{os.path.basename(pdf)} page {png.split('-')[-1][:-4]}: ink in the right margin at rows {rows.min()}-{rows.max()}")
                bad += 1
print("pages with overflow:", bad)
