#!/usr/bin/env python3
"""Run the commands of one guide chapter and record a terminal transcript for each step.

  record.py <chapter> [--mode venv|docker] [--repo DIR] [--only STEP ...] [--force]

A transcript (transcripts/<chapter>/<step>.txt) is the command as typed, prefixed with
"$ ", followed by its combined stdout/stderr and a closing line with the exit code and the
elapsed time.  Long outputs keep their first and last lines.  In docker mode every step runs
in a fresh container of the image `bi-certification` with results/, datasets/raw/,
checkpoints/ and external/ of the repository mounted, so the steps share their files.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GUIDE = HERE.parent
sys.path.insert(0, str(HERE))
from plans import PLANS  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
HEAD, TAIL = 12, 18
MAXCOL = 200


def shorten(lines, head=HEAD, tail=TAIL):
    if len(lines) <= head + tail + 5:
        return lines
    omitted = len(lines) - head - tail
    return lines[:head] + [f"[... {omitted} lines omitted ...]"] + lines[-tail:]


def clean(text: str, repo: str) -> str:
    text = ANSI.sub("", text).replace("\r\n", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[^\n]*\r", "", text)            # progress bars: keep the final state of each line
    # progress bars (tqdm): keep only the final 100% line of each bar
    kept = []
    keras = re.compile(r"^\s*(\d+)/(\d+) \[[=>.]*\]")
    for ln in text.split("\n"):
        if re.match(r"^\s*\d+%\|", ln) and not ln.lstrip().startswith("100%"):
            continue                       # tqdm bar, not final
        m = keras.match(ln)
        if m and m.group(1) != m.group(2):
            continue                       # Keras bar, not final
        kept.append(ln)
    text = "\n".join(kept)
    text = text.replace(repo + "/", "").replace(repo, ".")
    text = text.replace("/artifact/", "")
    return text


def run_step(cmd: str, mode: str, repo: str, timeout: int) -> tuple[str, int, float]:
    if mode == "host":
        argv = ["bash", "-c", f"cd '{repo}' && {cmd}"]
    elif mode == "venv":
        bindir = os.environ.get("GUIDE_PYTHON_BIN")      # a directory holding python/pip
        act = f"export PATH='{bindir}':$PATH" if bindir else "source .venv/bin/activate"
        argv = ["bash", "-c", f"cd '{repo}' && {act} && {cmd}"]
    else:
        mounts = []
        results_host = os.environ.get("GUIDE_RESULTS", f"{repo}/results")   # a different host folder for results/
        for sub, host in (("results", results_host), ("datasets/raw", f"{repo}/datasets/raw"),
                          ("checkpoints", f"{repo}/checkpoints"), ("external", f"{repo}/external")):
            mounts += ["-v", f"{host}:/artifact/{sub}"]
        argv = ["docker", "run", "--rm", *mounts, "bi-certification", "bash", "-c", cmd]
    t0 = time.time()
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, errors="replace")
        out, code = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out, code = (e.stdout or "") + (e.stderr or "") + "\n[timed out]", 124
    return out, code, time.time() - t0


def fmt_elapsed(s: float) -> str:
    s = int(round(s))
    return f"{s // 60} min {s % 60} s" if s >= 60 else f"{s} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chapter", choices=sorted(PLANS))
    ap.add_argument("--mode", choices=["venv", "docker", "host"], default="docker")
    ap.add_argument("--repo", default=os.environ.get("GUIDE_REPO", str(GUIDE.parents[1])), help="repository root (default: the one containing docs/guides)")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-dir", default="transcripts", help="directory under guide/ for the transcripts")
    a = ap.parse_args()
    repo = str(Path(a.repo).resolve())
    outdir = GUIDE / a.out_dir / a.chapter
    outdir.mkdir(parents=True, exist_ok=True)
    for step in PLANS[a.chapter]:
        name, cmd = step["name"], step["cmd"]
        if a.only and name not in a.only:
            continue
        target = outdir / f"{name}.txt"
        if target.exists() and not a.force:
            print(f"skip {name} (exists)")
            continue
        print(f"[{a.chapter}/{name}] running ...", flush=True)
        out, code, dt = run_step(cmd, step.get("mode", a.mode), repo, step.get("timeout", 7200))
        rawdir = GUIDE / "transcripts_raw" / a.chapter
        rawdir.mkdir(parents=True, exist_ok=True)
        (rawdir / f"{name}.log").write_text(out)
        lines = clean(out, repo).rstrip("\n").split("\n") if out.strip() else []
        shown = step.get("show", "both")
        if shown == "tail":
            n = step.get("n", 40)
            if len(lines) > n:
                key = re.compile(r"recomputed|^\[PASS\]|^\[DIFF\]|exit code \d|checksum|^window |Selected bytes|successes=|^\$ ")
                kept = [ln for ln in lines[:-n] if key.search(ln)]
                lines = ["[... earlier output omitted; key lines kept ...]"] + kept + ["[...]"] + lines[-n:]
        else:
            lines = shorten(lines, step.get("head", HEAD), step.get("tail", TAIL))
        lines = [ln if len(ln) <= MAXCOL else ln[:MAXCOL] + " …" for ln in lines]
        body = "$ " + cmd.replace(" && ", " && \\\n    ") + "\n" + "\n".join(lines)
        body += f"\n[exit code {code}, {fmt_elapsed(dt)}]\n"
        target.write_text(body)
        print(f"[{a.chapter}/{name}] exit {code} in {fmt_elapsed(dt)} -> {target.relative_to(GUIDE)}")
        if code != 0 and step.get("stop_on_error", True):
            print("stopping after a failed step"); break


if __name__ == "__main__":
    main()
