#!/usr/bin/env python3
"""Build the full-suite local-family table (Table 9) from relaxed-margin counts.

For every dataset and radius, the certificate takes the reference model with
the largest relaxed-margin count S_loc(eps) among the M = 80 TCHES20 models
and applies a one-sided KL-binomial upper endpoint with the confidence split
delta / (M * |grid|), as in `bi_scope_certificates.compute_full_suite_bi_loc`.

Counts are read from `paper_results/archived/full_suite_relaxed_margin_counts.csv`,
or from `<dataset>_full_suite_bi_loc_per_center.csv` files found under
`--recomputed-dir`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bi_scope_certificates.compute_bi_loc_margin_curves import bi_from_p, kl_upper
from paper_results.common import (
    ARCHIVED,
    DATASET_LABELS,
    DEFAULT_OUT,
    DELTA,
    EXPECTED,
    PaperCheck,
    read_rows,
    write_markdown,
    write_rows,
)

DATASETS = ["aes_rd", "ascad_desync_0", "ascad_desync_50", "ascad_desync_100"]
PAPER_RADII = [0.0, 1e-4, 3e-4, 1e-3, 1e-2]


def load_counts(path: Path) -> dict[str, list[dict[str, object]]]:
    by_dataset: dict[str, list[dict[str, object]]] = {}
    for row in read_rows(path):
        by_dataset.setdefault(row["dataset"], []).append(
            {
                "center_id": row["center_id"],
                "architecture": row["architecture"],
                "epsilon": float(row["epsilon"]),
                "n_te": int(row["n_te"]),
                "S_loc": int(row["S_loc"]),
            }
        )
    return by_dataset


def summarize(dataset: str, rows: list[dict[str, object]], delta: float) -> list[dict[str, object]]:
    centers = list(dict.fromkeys(str(row["center_id"]) for row in rows))
    grid = sorted({float(row["epsilon"]) for row in rows})
    sizes = {int(row["n_te"]) for row in rows}
    if len(sizes) != 1:
        raise ValueError(f"{dataset}: mixed attack-set sizes {sizes}")
    n_te = sizes.pop()
    alpha = delta / (len(centers) * len(grid))
    out = []
    for eps in grid:
        at_eps = [row for row in rows if float(row["epsilon"]) == eps]
        # First maximum in file order, matching the evaluator's tie handling.
        active = max(at_eps, key=lambda row: int(row["S_loc"]))
        s_loc = int(active["S_loc"])
        p_plus = kl_upper(s_loc, n_te, alpha)
        out.append(
            {
                "dataset": dataset,
                "M": len(centers),
                "epsilon": eps,
                "n_te": n_te,
                "S_loc": s_loc,
                "p_hat_loc": s_loc / n_te,
                "p_loc_plus": p_plus,
                "BI_loc_plus": bi_from_p(p_plus),
                "active_center": active["center_id"],
                "architecture": active["architecture"],
                "delta_effective": alpha,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recomputed-dir", type=Path, default=DEFAULT_OUT / "recomputed")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--delta", type=float, default=DELTA)
    args = parser.parse_args()

    archived = load_counts(ARCHIVED / "full_suite_relaxed_margin_counts.csv")
    table = []
    for dataset in DATASETS:
        rows, source = archived[dataset], "archived"
        if args.recomputed_dir.exists():
            matches = sorted(args.recomputed_dir.rglob(f"{dataset}_full_suite_bi_loc_per_center.csv"))
            if matches:
                rows, source = load_counts(matches[0])[dataset], "recomputed"
                reference = {(r["center_id"], r["epsilon"]): r["S_loc"] for r in archived[dataset]}
                same = sum(1 for r in rows if reference.get((r["center_id"], r["epsilon"])) == r["S_loc"])
                print(f"{dataset}: {same}/{len(rows)} recomputed (model, radius) counts equal the archived counts")
        for record in summarize(dataset, rows, args.delta):
            record["counts_source"] = source
            table.append(record)
    write_rows(args.out_dir / "table9_biloc_full_suite.csv", table)

    by_key = {(r["dataset"], r["epsilon"]): r for r in table}
    check = PaperCheck("Table 9 (full-suite local-family certificates)")
    for printed in read_rows(EXPECTED / "table9_biloc_full_suite.csv"):
        record = by_key[(printed["dataset"], float(printed["epsilon"]))]
        label = f"{DATASET_LABELS[printed['dataset']]} eps={float(printed['epsilon']):g}"
        # Relaxed counts sit on float32 score margins, so a recomputation on
        # other hardware may move a count by a few traces.
        slack = 0 if record["counts_source"] == "archived" else max(3, int(0.002 * int(printed["S_loc"])))
        if abs(int(record["S_loc"]) - int(printed["S_loc"])) <= slack:
            check.passed += 1
        else:
            check.failures.append(f"{label} S_loc: regenerated {record['S_loc']}, paper {printed['S_loc']}")
        bits_slack = 0.0 if record["counts_source"] == "archived" else 0.01
        check.number(f"{label} p_hat", float(record["p_hat_loc"]), printed["p_hat_loc"], slack=bits_slack / 10)
        check.number(f"{label} BI_loc_plus", float(record["BI_loc_plus"]), printed["BI_loc_plus"], slack=bits_slack)
        check.exact(f"{label} architecture", record["architecture"], printed["architecture"])
    d0_zero = by_key[("ascad_desync_0", 0.0)]
    check.number("ASCAD d0 local bound at eps=0 (Section 5.3)", float(d0_zero["BI_loc_plus"]), "1.709")

    body = [
        [
            DATASET_LABELS[r["dataset"]],
            f"{r['epsilon']:g}",
            f"{r['S_loc']:,}",
            f"{r['p_hat_loc']:.3f}",
            f"{r['BI_loc_plus']:.2f}",
            str(r["architecture"]),
            str(r["counts_source"]),
        ]
        for r in table
        if any(abs(float(r["epsilon"]) - eps) < 1e-12 for eps in PAPER_RADII)
    ]
    print()
    print(
        write_markdown(
            args.out_dir / "table9_biloc_full_suite.md",
            ["Benchmark", "eps", "S_loc", "p_hat_loc", "BI_loc_plus", "Arch.", "counts"],
            body,
            title=f"Table 9: full-suite local-family certificate over M = 80 models (delta = {args.delta:g})",
        )
    )
    raise SystemExit(0 if check.report() else 1)


if __name__ == "__main__":
    main()
