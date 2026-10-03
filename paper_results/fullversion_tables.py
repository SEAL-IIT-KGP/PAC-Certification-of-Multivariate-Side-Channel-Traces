#!/usr/bin/env python3
"""Check the two full-version tables quoted in Section 5 of the paper.

- Raw-ASCAD four-byte composition (`tab:raw-ascad-four-byte-composition`,
  quoted in Section 5.1): read from `multibyte_composition.compose_four_byte_recovery`
  outputs under `--recomputed-dir` (`multibyte_composition/.../multibyte_composition_summary*.csv`).
  The printed table came from byte attackers that were overwritten by a later
  training run (see `multibyte_composition/README.md`), so a rerun is compared
  with the printed numbers and with the archived final-run numbers, and the
  check is on the claim: the four-byte key ranks first in most campaigns from
  500 traces on, with median rank 0.
- PI/EPI under temperature rescaling (`tab:epi-temperature`, summarized in
  Section 5.2): read from `epi_comparison.epi_head_to_head` outputs
  (`epi_comparison/main/<dataset>/temperature_stress.csv`) under `--recomputed-dir`,
  or from the archived per-temperature values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from paper_results.common import (
    ARCHIVED,
    DATASET_LABELS,
    DEFAULT_OUT,
    EXPECTED,
    PaperCheck,
    read_rows,
    write_markdown,
    write_rows,
)


def four_byte(recomputed_dir: Path, out_dir: Path) -> bool:
    printed = {int(r["traces"]): r for r in read_rows(EXPECTED / "fullversion_four_byte.csv")}
    archived = {int(r["trace_count"]): r for r in read_rows(ARCHIVED / "four_byte_composition_1000_campaigns.csv")}
    candidates = (
        sorted(recomputed_dir.rglob("multibyte_composition_summary_250.csv"))
        or sorted(recomputed_dir.rglob("multibyte_composition_summary.csv"))
        if recomputed_dir.exists()
        else []
    )
    rerun = {int(r["trace_count"]): r for r in read_rows(candidates[0])} if candidates else {}
    check = PaperCheck("Four-byte composition (full version)")
    body = []
    rows = []
    for traces in sorted(printed):
        p = printed[traces]
        a = archived.get(traces)
        r = rerun.get(traces)
        rows.append(
            {
                "traces": traces,
                "printed_successes": f"{p['successes']}/{p['campaigns']}",
                "printed_rate": p["empirical_rate"],
                "archived_final_models_rate": a["empirical_full_success"] if a else "",
                "archived_final_models_campaigns": a["campaigns"] if a else "",
                "rerun_rate": r["empirical_full_success"] if r else "",
                "rerun_successes": f"{r['full_successes']}/{r['campaigns']}" if r else "",
                "rerun_median_rank": r["median_full_rank"] if r else "",
            }
        )
        body.append(
            [
                str(traces),
                rows[-1]["printed_successes"],
                p["empirical_rate"],
                rows[-1]["archived_final_models_rate"],
                rows[-1]["rerun_successes"],
                rows[-1]["rerun_rate"],
                rows[-1]["rerun_median_rank"],
            ]
        )
    write_rows(out_dir / "fullversion_four_byte.csv", rows)
    print(
        write_markdown(
            out_dir / "fullversion_four_byte.md",
            [
                "Traces",
                "Printed",
                "Printed rate",
                "Archived final models (1,000 campaigns)",
                "Rerun",
                "Rerun rate",
                "Rerun median rank",
            ],
            body,
            title="Full version: raw-ASCAD four-byte composition",
        )
    )
    if not rerun:
        print("No rerun found; run multibyte_composition (see its README) to recompute this table.")
        check.exact(
            "archived final-model rate at 500 traces exceeds the printed rate",
            float(archived[500]["empirical_full_success"]) >= float(printed[500]["empirical_rate"]),
            True,
        )
        return check.report()
    for traces in (500, 1000, 2000):
        r = rerun.get(traces)
        if r is None:
            check.failures.append(f"rerun has no row for {traces} traces")
            continue
        check.exact(f"median four-byte rank is 0 at {traces} traces", float(r["median_full_rank"]) == 0.0, True)
        check.exact(
            f"four-byte success at {traces} traces is at least the printed {printed[traces]['empirical_rate']}",
            float(r["empirical_full_success"]) >= float(printed[traces]["empirical_rate"]) - 0.05,
            True,
        )
    return check.report()


def epi_temperature(recomputed_dir: Path, out_dir: Path) -> bool:
    printed = {r["dataset"]: r for r in read_rows(EXPECTED / "fullversion_epi_temperature.csv")}
    source = "archived"
    by_dataset: dict[str, list[dict[str, str]]] = {}
    fresh = sorted(recomputed_dir.rglob("temperature_stress.csv")) if recomputed_dir.exists() else []
    if fresh:
        source = "recomputed"
        for path in fresh:
            by_dataset[path.parent.name] = read_rows(path)
    else:
        for row in read_rows(ARCHIVED / "epi_temperature_stress.csv"):
            by_dataset.setdefault(row["dataset"], []).append(row)
    check = PaperCheck("PI/EPI temperature sweep (full version)")
    rows, body = [], []
    for dataset in printed:
        if dataset not in by_dataset:
            check.failures.append(f"{dataset}: no temperature sweep found")
            continue
        temps = by_dataset[dataset]
        raw = np.array([float(r["raw_pi_bits"]) for r in temps])
        epi = np.array([float(r["epi_bits"]) for r in temps])
        preserved = all(str(r["argmax_preserved"]).lower() == "true" for r in temps)
        rows.append(
            {
                "dataset": dataset,
                "temperatures": len(temps),
                "raw_pi_variation": float(np.ptp(raw)),
                "epi_variation": float(np.ptp(epi)),
                "argmax_preserved": preserved,
                "source": source,
            }
        )
        body.append(
            [DATASET_LABELS[dataset], f"{np.ptp(raw):.2f}", f"{np.ptp(epi):.2e}", "yes" if preserved else "no", source]
        )
        label = DATASET_LABELS[dataset]
        # Raw PI spans depend on the selected model's scores; a rerun on other hardware may move them slightly.
        check.number(
            f"{label} raw PI variation",
            float(np.ptp(raw)),
            printed[dataset]["raw_pi_variation"],
            slack=0.0 if source == "archived" else 0.05,
        )
        check.exact(f"{label} EPI variation below 1e-8 bits", float(np.ptp(epi)) < 1e-8, True)
        check.exact(f"{label} predicted labels preserved at every temperature", preserved, True)
    write_rows(out_dir / "fullversion_epi_temperature.csv", rows)
    print(
        write_markdown(
            out_dir / "fullversion_epi_temperature.md",
            ["Benchmark", "Raw PI variation", "EPI variation", "Labels preserved", "source"],
            body,
            title="Full version: variation across the temperature sweep (bits)",
        )
    )
    return check.report()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recomputed-dir", type=Path, default=DEFAULT_OUT / "recomputed")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ok_a = four_byte(args.recomputed_dir, args.out_dir)
    ok_b = epi_temperature(args.recomputed_dir, args.out_dir)
    raise SystemExit(0 if ok_a and ok_b else 1)


if __name__ == "__main__":
    main()
