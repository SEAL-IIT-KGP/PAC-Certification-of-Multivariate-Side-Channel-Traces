#!/usr/bin/env python3
"""Reproduce the paper's tables and figures.

    python -m paper_results.reproduce archived
        Rebuild Tables 7-9 and Figures 1-2 from the archived per-model counts
        (no downloads, about one minute) and compare every value with the paper.

    python -m paper_results.reproduce tches20 [--quick]
        Recompute, from the public TCHES20 traces and pretrained models, the
        attack success counts of all 480 models (Table 7, six rows), the
        relaxed-margin counts of the four 80-model suites (Table 9), EPI and key
        recovery for the selected models (Tables 7 and 8), and the six public
        reference models of Figure 2. `--quick` evaluates only the six selected
        models. Run `paper_results/fetch_tches20.sh` first.

    python -m paper_results.reproduce estranet --estranet-root ... (see paper_results/README.md)
        Recompute the two EstraNet rows of Table 7 from the released checkpoints.

    python -m paper_results.reproduce fullversion
        Recompute the two full-version tables quoted in Section 5 from the public
        TCHES20 models (PI/EPI temperature sweep) and the raw ASCAD traces
        (four-byte composition; 6 GB download, PyTorch).

    python -m paper_results.reproduce report
        Rebuild the tables and figures from whatever has been recomputed so far.

Recomputed counts are written to `results/paper_results/recomputed/`; the
tables and figures are written to `results/paper_results/`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from paper_results.common import DEFAULT_OUT, REPO, TCHES20_DATASETS

FULL_SUITE_LOC_DATASETS = ["aes_rd", "ascad_desync_0", "ascad_desync_50", "ascad_desync_100"]
PUBLIC_CENTER_DATASETS = ["ascad_desync_50", "ascad_desync_100", "aes_rd"]
BUILDERS = [
    ("Table 7", "paper_results.table7_bi_suite"),
    ("Table 8", "paper_results.table8_key_recovery"),
    ("Table 9", "paper_results.table9_biloc_full_suite"),
    ("Figure 1", "paper_results.figure1_dimension_sweep"),
    ("Figure 2", "paper_results.figure2_biloc_selected_centers"),
    ("Full-version tables", "paper_results.fullversion_tables"),
]


def run(module: str, *args: object, check: bool = True) -> int:
    command = [sys.executable, "-m", module, *[str(a) for a in args]]
    print(f"\n$ python -m {module} {' '.join(str(a) for a in args)}", flush=True)
    started = time.time()
    code = subprocess.call(command, cwd=REPO)
    print(f"[{module}: {time.time() - started:.0f} s, exit code {code}]", flush=True)
    if check and code != 0:
        raise SystemExit(f"{module} failed")
    return code


def report(out_dir: Path) -> int:
    recomputed = out_dir / "recomputed"
    status = []
    for name, module in BUILDERS:
        code = run(module, "--recomputed-dir", recomputed, "--out-dir", out_dir, check=False)
        status.append((name, code))
    print("\n================ summary ================")
    for name, code in status:
        print(f"{name:<9} {'matches the paper' if code == 0 else 'differs from the paper (see above)'}")
    print(f"tables and figures: {out_dir}")
    return 0 if all(code == 0 for _, code in status) else 1


def gaussian_oracle(out_dir: Path) -> None:
    from paper_results.common import PaperCheck, read_rows

    oracle_dir = out_dir / "gaussian_oracle"
    run("bayes_oracle.gaussian_oracle_suite", "--output-dir", oracle_dir)
    rows = {row["model"]: row for row in read_rows(oracle_dir / "gaussian_oracle_suite_summary.csv")}
    check = PaperCheck("Synthetic Gaussian oracle (full version, Bayes-reference table)")
    check.number("Bayes success", float(rows["bayes_oracle_known_means"]["p_suite_mean"]), "0.989577")
    check.number("suite success", float(rows["learned_gaussian_template"]["p_suite_mean"]), "0.989503")
    check.report()


def tches20(args: argparse.Namespace) -> None:
    root = args.tches_root
    models = root / "models" / "pretrained_models" / "models"
    if not models.exists():
        raise SystemExit(f"{models} not found; run paper_results/fetch_tches20.sh first")
    recomputed = args.out_dir / "recomputed"
    datasets = args.datasets or TCHES20_DATASETS

    if not args.quick:
        suite_dir = recomputed / "tches20_suite"
        for dataset in datasets:
            if (suite_dir / f"{dataset}_attack_success.csv").exists() and not args.force:
                print(f"{dataset}: suite counts already recomputed")
                continue
            extra = ["--max-eval-traces", 12500] if dataset == "aes_rd" else []
            run(
                "bi_suite_8datasets.evaluate_tches20_pretrained_fixed_split",
                "--dataset",
                dataset,
                "--split",
                "attack",
                *extra,
                "--models-dir",
                models,
                "--dataset-dir",
                root / "datasets",
                "--tches20-src-dir",
                root / "src",
                "--output-dir",
                suite_dir,
            )
        # Suite endpoints (BI_suite,+) are an input of the local-family evaluator.
        run("paper_results.table7_bi_suite", "--recomputed-dir", recomputed, "--out-dir", args.out_dir, check=False)
        loc_dir = recomputed / "tches20_full_suite_loc"
        for dataset in [d for d in FULL_SUITE_LOC_DATASETS if d in datasets]:
            if (loc_dir / f"{dataset}_full_suite_bi_loc_per_center.csv").exists() and not args.force:
                print(f"{dataset}: full-suite relaxed-margin counts already recomputed")
                continue
            run(
                "bi_scope_certificates.compute_full_suite_bi_loc",
                "--dataset",
                dataset,
                "--tches-root",
                root,
                "--suite-csv",
                args.out_dir / "bi_suite_dataset_summary.csv",
                "--output-dir",
                loc_dir,
            )
        for dataset in [d for d in PUBLIC_CENTER_DATASETS if d in datasets]:
            center_dir = recomputed / "selected_centers" / dataset
            if (center_dir / "epsilon_margin_counts.csv").exists() and not args.force:
                print(f"{dataset}: selected reference models already profiled")
                continue
            run(
                "bi_scope_certificates.profile_epsilon_calibration",
                "--dataset",
                dataset,
                "--public-only",
                "--tches-root",
                root,
                "--output-dir",
                center_dir,
            )

    run(
        "paper_results.evaluate_selected_models", "--tches-root", root, "--out-dir", recomputed, "--datasets", *datasets
    )


def fullversion(args: argparse.Namespace) -> None:
    root = args.tches_root
    recomputed = args.out_dir / "recomputed"
    for index, dataset in enumerate(TCHES20_DATASETS):
        if (recomputed / "epi_comparison" / "main" / dataset / "result.json").exists() and not args.force:
            print(f"{dataset}: EPI comparison already recomputed")
            continue
        run(
            "epi_comparison.epi_head_to_head",
            "--dataset",
            dataset,
            "--dataset-dir",
            root / "datasets",
            "--tches20-src-dir",
            root / "src",
            "--models-dir",
            root / "models",
            "--output-dir",
            recomputed / "epi_comparison" / "main",
        )
    raw = REPO / "datasets/raw/ascad_raw/ATMega8515_raw_traces.h5"
    if not raw.exists():
        run("multibyte_composition.fetch_raw_ascad")
    scan = recomputed / "multibyte_composition" / "poi_scan"
    attackers = recomputed / "multibyte_composition" / "byte_attackers"
    if not (scan / "raw_ascad_poi_scan_topk.csv").exists() or args.force:
        run("multibyte_composition.scan_raw_ascad_pois", "--raw-h5", raw, "--output-dir", scan)
    for byte in (0, 1, 2, 3, 4, 5, 8, 15):
        if (attackers / f"byte_{byte:02d}_posteriors.npz").exists() and not args.force:
            continue
        run(
            "multibyte_composition.train_raw_ascad_byte_attacker",
            "--target-byte",
            byte,
            "--raw-h5",
            raw,
            "--scan-topk",
            scan / "raw_ascad_poi_scan_topk.csv",
            "--output-dir",
            attackers,
        )
    run(
        "multibyte_composition.compose_four_byte_recovery",
        "--input-dir",
        attackers,
        "--campaigns",
        250,
        "--trace-counts",
        "1,2,5,10,20,50,100,200,500,1000,2000",
    )
    (attackers / "multibyte_composition_summary.csv").replace(attackers / "multibyte_composition_summary_250.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("level", choices=["archived", "tches20", "estranet", "fullversion", "report"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tches-root", type=Path, default=REPO / "datasets/raw/tches20/TCHES20V3_CNN_SCA")
    parser.add_argument(
        "--datasets", nargs="+", choices=TCHES20_DATASETS, help="Restrict `tches20` to these benchmarks."
    )
    parser.add_argument("--quick", action="store_true", help="`tches20`: evaluate only the six selected models.")
    parser.add_argument("--force", action="store_true", help="Recompute counts that already exist.")
    parser.add_argument(
        "--skip-oracle", action="store_true", help="`archived`: skip the synthetic Gaussian oracle run."
    )
    args, extra = parser.parse_known_args()

    if args.level == "archived":
        if not args.skip_oracle:
            gaussian_oracle(args.out_dir)
        # The archived level ignores recomputed counts.
        status = []
        for name, module in BUILDERS:
            status.append(
                (name, run(module, "--recomputed-dir", args.out_dir / "none", "--out-dir", args.out_dir, check=False))
            )
        print("\n================ summary ================")
        for name, code in status:
            print(f"{name:<9} {'matches the paper' if code == 0 else 'differs from the paper (see above)'}")
        print(f"tables and figures: {args.out_dir}")
        raise SystemExit(0 if all(code == 0 for _, code in status) else 1)
    if args.level == "tches20":
        tches20(args)
    elif args.level == "estranet":
        run("paper_results.evaluate_estranet_suites", "--out-dir", args.out_dir / "recomputed", *extra)
    elif args.level == "fullversion":
        fullversion(args)
    raise SystemExit(report(args.out_dir))


if __name__ == "__main__":
    main()
