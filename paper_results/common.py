"""Shared paths, labels, and paper-value checks for the table and figure builders."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
ARCHIVED = REPO / "paper_results" / "archived"
EXPECTED = REPO / "paper_results" / "expected"
DEFAULT_OUT = REPO / "results" / "paper_results"

DELTA = 1e-6
EPS_GRID = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]

DATASET_ORDER = [
    "ascad_desync_0",
    "ascad_desync_50",
    "ascad_desync_100",
    "ascad_random_key",
    "aes_hd",
    "aes_rd",
    "dpav4",
    "ches_ctf_2025",
]
TCHES20_DATASETS = ["ascad_desync_0", "ascad_desync_50", "ascad_desync_100", "aes_hd", "aes_rd", "dpav4"]
ESTRANET_DATASETS = ["ascad_random_key", "ches_ctf_2025"]

DATASET_LABELS = {
    "ascad_desync_0": "ASCAD d0",
    "ascad_desync_50": "ASCAD d50",
    "ascad_desync_100": "ASCAD d100",
    "ascad_random_key": "ASCAD random",
    "aes_hd": "AES-HD",
    "aes_rd": "AES-RD",
    "dpav4": "DPAv4",
    "ches_ctf_2025": "CHES-CTF-2025",
}
# Trace dimension d of the model input reported in the BI-suite table.
TRACE_DIMENSION = {
    "ascad_desync_0": 700,
    "ascad_desync_50": 700,
    "ascad_desync_100": 700,
    "ascad_random_key": 1400,
    "aes_hd": 1250,
    "aes_rd": 3500,
    "dpav4": 4000,
    "ches_ctf_2025": 3000,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, header: list[str], body: list[list[str]], title: str = "") -> str:
    lines = []
    if title:
        lines += [f"### {title}", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    lines += ["| " + " | ".join(row) + " |" for row in body]
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


class PaperCheck:
    """Collect comparisons between regenerated values and the values printed in the paper."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.passed = 0
        self.failures: list[str] = []

    def number(self, label: str, value: float, printed: str, slack: float = 0.0) -> None:
        """`printed` is the paper's string; the value must round to it (plus optional slack)."""
        text = printed.strip().replace(",", "")
        decimals = len(text.split(".")[1]) if "." in text else 0
        tolerance = 0.5 * 10.0 ** (-decimals) + 1e-9 + slack
        if abs(float(value) - float(text)) <= tolerance:
            self.passed += 1
        else:
            self.failures.append(f"{label}: regenerated {value:.6g}, paper {printed}")

    def exact(self, label: str, value: Any, printed: Any) -> None:
        if str(value) == str(printed):
            self.passed += 1
        else:
            self.failures.append(f"{label}: regenerated {value}, paper {printed}")

    def report(self) -> bool:
        status = "PASS" if not self.failures else "DIFF"
        print(f"[{status}] {self.name}: {self.passed} values match the paper, {len(self.failures)} differ")
        for failure in self.failures:
            print(f"    {failure}")
        return not self.failures
