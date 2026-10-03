#!/usr/bin/env python3
"""Build the key-byte recovery table (Table 8) for the selected suite members.

Each entry is the percentage of 100 shuffled trace orders of the fixed attack
pool in which the correct key byte ranks first after accumulating the model's
log-probabilities over the given number of traces.

Campaign counts are read from `paper_results/archived/selected_model_key_recovery.csv`,
or from `selected_models/<dataset>/summary.json` files written by
`paper_results.evaluate_selected_models` under `--recomputed-dir`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_results.common import (
    ARCHIVED,
    DATASET_LABELS,
    DATASET_ORDER,
    DEFAULT_OUT,
    EXPECTED,
    PaperCheck,
    read_rows,
    write_markdown,
    write_rows,
)

BUDGETS = [100, 500, 1000, 2000]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recomputed-dir", type=Path, default=DEFAULT_OUT / "recomputed")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    # (dataset, traces) -> (successes, campaigns, source)
    counts: dict[tuple[str, int], tuple[int, int, str]] = {}
    models: dict[str, str] = {}
    for row in read_rows(ARCHIVED / "selected_model_key_recovery.csv"):
        counts[(row["dataset"], int(row["traces"]))] = (int(row["successes"]), int(row["campaigns"]), "archived")
        models[row["dataset"]] = row["model_id"]
    if args.recomputed_dir.exists():
        for summary in sorted(args.recomputed_dir.rglob("selected_models/*/summary.json")):
            result = json.loads(summary.read_text())
            models[result["dataset"]] = result["model"]
            for key in [k for k in counts if k[0] == result["dataset"]]:
                del counts[key]
            for curve in result["keyrank"]:
                counts[(result["dataset"], int(curve["traces"]))] = (
                    int(curve["successes"]),
                    int(curve["campaigns"]),
                    "recomputed",
                )

    table = []
    for dataset in DATASET_ORDER:
        record: dict[str, object] = {"dataset": dataset, "model_id": models[dataset]}
        for budget in BUDGETS:
            entry = counts.get((dataset, budget))
            record[f"traces_{budget}"] = "" if entry is None else round(100.0 * entry[0] / entry[1])
        record["source"] = next(v[2] for k, v in counts.items() if k[0] == dataset)
        table.append(record)
    write_rows(args.out_dir / "table8_key_recovery.csv", table)

    by_dataset = {r["dataset"]: r for r in table}
    check = PaperCheck("Table 8 (key-byte recovery success)")
    for printed in read_rows(EXPECTED / "table8_key_recovery.csv"):
        for budget in BUDGETS:
            check.exact(
                f"{DATASET_LABELS[printed['dataset']]} at {budget} traces",
                by_dataset[printed["dataset"]][f"traces_{budget}"],
                printed[f"traces_{budget}"],
            )

    body = [
        [DATASET_LABELS[r["dataset"]]]
        + [("---" if r[f"traces_{b}"] == "" else str(r[f"traces_{b}"])) for b in BUDGETS]
        + [str(r["model_id"]), str(r["source"])]
        for r in table
    ]
    print(
        write_markdown(
            args.out_dir / "table8_key_recovery.md",
            ["Benchmark", "100", "500", "1,000", "2,000", "model", "source"],
            body,
            title="Table 8: key-byte recovery success (%) over 100 shuffled orders",
        )
    )
    print("CHES-CTF-2025 is listed for completeness; the paper's table reports the other seven benchmarks.")
    raise SystemExit(0 if check.report() else 1)


if __name__ == "__main__":
    main()
