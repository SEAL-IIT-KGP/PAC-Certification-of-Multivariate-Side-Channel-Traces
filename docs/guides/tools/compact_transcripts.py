#!/usr/bin/env python3
"""Shorten transcripts for the guides: keep the command, the first HEAD and last TAIL output
lines, cut lines longer than MAXCOL characters, keep the closing exit-code line."""
import re, sys
from pathlib import Path
HEAD, TAIL, MAXCOL = 12, 18, 200
KEY = re.compile(r"recomputed|^\[PASS\]|^\[DIFF\]|checksum|^window |Selected bytes|successes=")

def compact(path: Path) -> None:
    lines = path.read_text().rstrip("\n").split("\n")
    if not lines or not lines[0].startswith("$ "):
        return
    i = 1
    while i < len(lines) and lines[i].startswith("    ") and lines[i - 1].rstrip().endswith("\\"):
        i += 1
    cmd, body, closing = lines[:i], lines[i:], []
    if body and body[-1].startswith("[exit code"):
        closing, body = [body[-1]], body[:-1]
    body = [ln if len(ln) <= MAXCOL else ln[:MAXCOL] + " …" for ln in body]
    if len(body) > HEAD + TAIL + 4:
        middle_keys = [ln for ln in body[HEAD:-TAIL] if KEY.search(ln) and not ln.startswith("[...")]
        body = body[:HEAD] + [f"[... {len(body) - HEAD - TAIL} lines omitted ...]"] + middle_keys[:20] + body[-TAIL:]
    path.write_text("\n".join(cmd + body + closing) + "\n")

for p in sys.argv[1:] or sorted(Path("transcripts").glob("*/*.txt")):
    compact(Path(p))
print("compacted")
