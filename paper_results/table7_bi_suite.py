#!/usr/bin/env python3
"""Build the finite-suite BI table (Table 7) from per-model attack success counts.

The certificate arithmetic is the one used by the suite evaluator
(`bi_suite_8datasets.evaluate_tches20_pretrained_fixed_split.summarize_counts`):
a two-sided KL-binomial interval with the confidence split delta / (2M).

Counts are read from `paper_results/archived/suite_success_counts.csv`. When a
directory with freshly recomputed counts is given (`--recomputed-dir`), those
counts are used for every dataset found there and compared, model by model,
with the archived ones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bi_suite_8datasets.evaluate_tches20_pretrained_fixed_split import summarize_counts
from paper_results.common import (
    ARCHIVED,
    DATASET_LABELS,
    DATASET_ORDER,
    DEFAULT_OUT,
    DELTA,
    EXPECTED,
    TRACE_DIMENSION,
    PaperCheck,
    read_rows,
    write_markdown,
    write_rows,
)


def archived_counts() -> dict[str, list[dict[str, object]]]:
    by_dataset: dict[str, list[dict[str, object]]] = {}
    for row in read_rows(ARCHIVED / "suite_success_counts.csv"):
        by_dataset.setdefault(row["dataset"], []).append(
            {
                "dataset": row["dataset"],
                "split": "attack",
                "model_id": row["model_id"],
                "n_te": int(row["n_te"]),
                "successes": int(row["successes"]),
            }
        )
    return by_dataset


def recomputed_counts(directory: Path | None) -> dict[str, list[dict[str, object]]]:
    """Read `<dataset>_attack_success.csv` (TCHES20) and `<dataset>_estranet_success.csv` files."""
    found: dict[str, list[dict[str, object]]] = {}
    if directory is None or not directory.exists():
        return found
    for dataset in DATASET_ORDER:
        for name in (f"{dataset}_attack_success.csv", f"{dataset}_estranet_success.csv"):
            matches = sorted(directory.rglob(name))
            if not matches:
                continue
            rows = []
            for row in read_rows(matches[0]):
                if row.get("status", "complete") != "complete":
                    raise RuntimeError(f"{matches[0]}: model {row['model_id']} did not complete")
                rows.append(
                    {
                        "dataset": dataset,
                        "split": "attack",
                        "model_id": row["model_id"],
                        "n_te": int(row["n_te"]),
                        "successes": int(row["successes"]),
                    }
                )
            found[dataset] = rows
            break
    return found


def read_epi(recomputed_dir: Path | None) -> dict[str, tuple[float, str]]:
    epi = {row["dataset"]: (float(row["EPI"]), "archived") for row in read_rows(ARCHIVED / "selected_model_epi.csv")}
    if recomputed_dir is not None and recomputed_dir.exists():
        for summary in sorted(recomputed_dir.rglob("selected_models/*/summary.json")):
            result = json.loads(summary.read_text())
            epi[result["dataset"]] = (float(result["EPI"]), "recomputed")
    return epi


def compare_counts(fresh: list[dict[str, object]], archived: list[dict[str, object]]) -> tuple[int, int]:
    reference = {str(row["model_id"]): int(row["successes"]) for row in archived}
    same = sum(1 for row in fresh if reference.get(str(row["model_id"])) == int(row["successes"]))
    return same, len(fresh)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recomputed-dir", type=Path, default=DEFAULT_OUT / "recomputed")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--delta", type=float, default=DELTA)
    args = parser.parse_args()

    archived = archived_counts()
    fresh = recomputed_counts(args.recomputed_dir)
    epi = read_epi(args.recomputed_dir)
    expected = {row["dataset"]: row for row in read_rows(EXPECTED / "table7_bi_suite.csv")}
    check = PaperCheck("Table 7 (finite-suite certificates)")

    table = []
    for dataset in DATASET_ORDER:
        source = "recomputed" if dataset in fresh else "archived"
        rows = fresh.get(dataset, archived[dataset])
        summary = summarize_counts(rows, args.delta)
        epi_value, epi_source = epi[dataset]
        record = {
            "dataset": dataset,
            "d": TRACE_DIMENSION[dataset],
            "M": summary["M"],
            "n_te": summary["n_te"],
            "successes": max(int(row["successes"]) for row in rows),
            "p_obs": summary["p_obs"],
            "p_minus": summary["p_minus"],
            "p_plus": summary["p_plus"],
            "BI_obs": summary["BI_obs"],
            "BI_minus": summary["BI_minus"],
            "BI_plus": summary["BI_plus"],
            "R_margin": summary["R_margin"],
            "EPI": epi_value,
            "best_model": summary["best_model"],
            "counts_source": source,
            "EPI_source": epi_source,
        }
        if dataset in fresh:
            same, total = compare_counts(rows, archived[dataset])
            record["models_matching_archived_counts"] = f"{same}/{total}"
            print(f"{dataset}: {same}/{total} recomputed per-model counts equal the archived counts")
        else:
            record["models_matching_archived_counts"] = ""
        table.append(record)

        printed = expected[dataset]
        label = DATASET_LABELS[dataset]
        check.exact(f"{label} M", record["M"], printed["M"])
        check.exact(f"{label} n_te", record["n_te"], printed["n_te"])
        for field in ("p_obs", "p_minus", "p_plus", "BI_obs", "BI_minus", "BI_plus", "R_margin", "EPI"):
            check.number(f"{label} {field}", float(record[field]), printed[field])

    write_rows(args.out_dir / "table7_bi_suite.csv", table)
    # Suite endpoints in the layout read by bi_scope_certificates (--suite-csv).
    write_rows(
        args.out_dir / "bi_suite_dataset_summary.csv",
        [
            {"dataset": r["dataset"], "M": r["M"], "n_te": r["n_te"], "p_plus": r["p_plus"], "BI_plus": r["BI_plus"]}
            for r in table
        ],
    )
    body = [
        [
            DATASET_LABELS[r["dataset"]],
            f"{r['d']:,}",
            str(r["M"]),
            f"{r['n_te']:,}",
            f"{r['p_obs']:.5f}",
            f"[{r['p_minus']:.5f}, {r['p_plus']:.5f}]",
            f"{r['BI_obs']:.3f}",
            f"[{r['BI_minus']:.3f}, {r['BI_plus']:.3f}]",
            f"{r['R_margin']:.2f}",
            f"{r['EPI']:.3f}",
            r["counts_source"],
        ]
        for r in table
    ]
    print()
    print(
        write_markdown(
            args.out_dir / "table7_bi_suite.md",
            ["Dataset", "d", "M", "n_te", "p_obs", "[p-, p+]", "BI_obs", "[BI-, BI+]", "R_margin", "EPI", "counts"],
            body,
            title=f"Table 7: finite-suite certificates (delta = {args.delta:g})",
        )
    )
    raise SystemExit(0 if check.report() else 1)


if __name__ == "__main__":
    main()
