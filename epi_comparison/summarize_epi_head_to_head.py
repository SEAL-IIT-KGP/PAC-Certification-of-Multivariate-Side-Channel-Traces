#!/usr/bin/env python3
"""Validate and summarize native-dimension BI versus IUH+22 EPI cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(directory: Path) -> None:
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(None, 1)
        path = directory / name.strip()
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(f"checksum mismatch {path}: {actual} != {expected}")


def wilson95(successes: int, total: int) -> tuple[float, float]:
    """Wilson score interval for the model's held-out hard-label accuracy."""
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - radius, center + radius


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    scarcity_rows = []
    temp_rows = []
    for result_path in sorted(args.input_dir.glob("*/result.json")):
        directory = result_path.parent
        verify(directory)
        result = json.loads(result_path.read_text())
        if not result["selection_calibration_evaluation_disjoint"]:
            raise AssertionError(f"non-disjoint split in {result_path}")
        successes = round(result["evaluation_accuracy"] * result["evaluation_records"])
        result["evaluation_successes"] = successes
        result["evaluation_accuracy_wilson95"] = wilson95(successes, result["evaluation_records"])
        result["accuracy_wilson_above_chance"] = result["evaluation_accuracy_wilson95"][0] > 1.0 / result["classes"]
        results.append(result)
        with (directory / "calibration_scarcity.csv").open() as handle:
            for row in csv.DictReader(handle):
                scarcity_rows.append(
                    {"dataset": result["dataset"], "native_dimension": result["native_dimension"], **row}
                )
        with (directory / "temperature_stress.csv").open() as handle:
            for row in csv.DictReader(handle):
                temp_rows.append({"dataset": result["dataset"], "native_dimension": result["native_dimension"], **row})

    if not results:
        raise FileNotFoundError(f"no result.json cells below {args.input_dir}")

    fields = [
        "dataset",
        "native_dimension",
        "suite_size",
        "selected_model",
        "selection_records",
        "calibration_records",
        "evaluation_records",
        "evaluation_successes",
        "evaluation_accuracy",
        "evaluation_accuracy_wilson95",
        "accuracy_wilson_above_chance",
        "finite_benchmark_bi_lower_bits",
        "raw_pi_bits",
        "epi_beta_hat",
        "epi_bits",
        "raw_pi_ci95",
        "epi_ci95",
    ]
    summary_csv = args.output_dir / "summary.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=lambda row: row["native_dimension"]):
            writer.writerow({key: result[key] for key in fields})

    for path, rows in (
        (args.output_dir / "scarcity_all.csv", scarcity_rows),
        (args.output_dir / "temperature_all.csv", temp_rows),
    ):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# BI versus IUH+22 EPI at native long dimensions",
        "",
        "Selection, beta calibration, and evaluation are class-balanced and mutually disjoint. BI and EPI use the same selected frozen model. BI is an exact finite-benchmark min-entropy lower bound; EPI is held-out Shannon cross-information, so their bit magnitudes are not ranked as interchangeable quantities.",
        "",
        "| Dataset | Native d | Eval n | Accuracy (Wilson 95%) | finite BI | raw PI | EPI | beta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(results, key=lambda item: item["native_dimension"]):
        lines.append(
            f"| {row['dataset']} | {row['native_dimension']} | {row['evaluation_records']} | "
            f"{row['evaluation_accuracy']:.6f} [{row['evaluation_accuracy_wilson95'][0]:.6f}, "
            f"{row['evaluation_accuracy_wilson95'][1]:.6f}] | {row['finite_benchmark_bi_lower_bits']:.4f} | "
            f"{row['raw_pi_bits']:.4f} | {row['epi_bits']:.4f} | {row['epi_beta_hat']:.4g} |"
        )

    temp_spans, scarcity_spans = [], []
    for row in results:
        directory = args.input_dir / row["dataset"]
        temp = list(csv.DictReader((directory / "temperature_stress.csv").open()))
        epi = np.asarray([float(x["epi_bits"]) for x in temp])
        raw = np.asarray([float(x["raw_pi_bits"]) for x in temp])
        if not all(x["argmax_preserved"].lower() == "true" for x in temp):
            raise AssertionError(f"temperature changed argmax for {row['dataset']}")
        temp_spans.append((row["dataset"], float(np.ptp(raw)), float(np.ptp(epi))))
        scarcity = list(csv.DictReader((directory / "calibration_scarcity.csv").open()))
        epi_mean = np.asarray([float(x["epi_mean_bits"]) for x in scarcity])
        scarcity_spans.append(
            (row["dataset"], float(np.ptp(epi_mean)), max(float(x["epi_std_bits"]) for x in scarcity))
        )

    lines.extend(
        [
            "",
            "## Calibration and temperature findings",
            "",
            "| Dataset | raw-PI temperature span | optimized-EPI span | scarcity mean span | max scarcity SD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    by_scarcity = {name: (span, sd) for name, span, sd in scarcity_spans}
    for name, raw_span, epi_span in temp_spans:
        scarcity_span, scarcity_sd = by_scarcity[name]
        lines.append(f"| {name} | {raw_span:.6g} | {epi_span:.6g} | {scarcity_span:.6g} | {scarcity_sd:.6g} |")

    positive_bi_low_epi = [
        row["dataset"] for row in results if row["finite_benchmark_bi_lower_bits"] > 0 and row["epi_ci95"][0] <= 0
    ]
    finite_bi_numerically_exceeds_epi = [
        row["dataset"]
        for row in results
        if row["finite_benchmark_bi_lower_bits"] > row["epi_bits"] and row["accuracy_wilson_above_chance"]
    ]
    lines.extend(
        [
            "",
            "## Bounded conclusion",
            "",
            f"Datasets with a positive finite BI lower bound but an EPI interval touching zero: {', '.join(positive_bi_low_epi) if positive_bi_low_epi else 'none'}.",
            "",
            "Datasets where finite-benchmark BI is numerically above EPI and the hard-label accuracy's Wilson 95% lower endpoint is above random guessing: "
            f"{', '.join(finite_bi_numerically_exceeds_epi) if finite_bi_numerically_exceeds_epi else 'none'}.",
            "",
            "Dimension by itself is not an EPI failure mode after model scores exist: EPI optimizes one scalar beta. BI is better only in the narrower sense supported by the artifacts: it is calibration-free and is an exact finite-benchmark certificate, whereas EPI requires probability scores, a calibration protocol, and reports a Shannon-information estimate. Any stronger claim requires evidence of EPI instability or loss of operational alignment on the same frozen model.",
        ]
    )
    report = args.output_dir / "REPORT.md"
    report.write_text("\n".join(lines) + "\n")

    checksum = args.output_dir / "SHA256SUMS"
    primary = [summary_csv, args.output_dir / "scarcity_all.csv", args.output_dir / "temperature_all.csv", report]
    checksum.write_text("".join(f"{sha256(path)}  {path.name}\n" for path in primary))
    print(report.read_text())


if __name__ == "__main__":
    main()
