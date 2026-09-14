#!/usr/bin/env python3
"""Build concrete audited-radius BI_loc points for the Figure 2 sweep."""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import pandas as pd


DATASET_LABELS = {
    "ascad_desync_50": "ASCAD d50",
    "ascad_desync_100": "ASCAD d100",
    "ascad_random_key": "ASCAD random",
    "aes_rd": "AES-RD",
}

DATASET_ORDER = {
    "ascad_desync_50": 0,
    "ascad_desync_100": 1,
    "ascad_random_key": 2,
    "aes_rd": 3,
}
ARCH_ORDER = {"MLP": 0, "CNN": 1, "Transformer": 2}


def esc(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def fmt_float(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def fmt_bits(value: float) -> str:
    return f"{float(value):.3f}"


def fmt_eps(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    exponent = f"{value:.0e}"
    if exponent == "1e-04":
        return r"10^{-4}"
    return exponent.replace("e-0", r"\times 10^{-").replace("e-", r"\times 10^{-") + "}"


def sort_points(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_dataset_order"] = out["dataset"].map(DATASET_ORDER).fillna(99)
    out["_arch_order"] = out["architecture"].map(ARCH_ORDER).fillna(99)
    out = out.sort_values(["_dataset_order", "dataset", "_arch_order", "architecture"])
    return out.drop(columns=["_dataset_order", "_arch_order"])


def nearest_epsilon_rows(curves: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    rows = []
    for key, group in curves.groupby(["dataset", "architecture", "center_id"], sort=False):
        idx = (group["epsilon"].astype(float) - float(epsilon)).abs().idxmin()
        row = group.loc[idx].copy()
        if not math.isclose(float(row["epsilon"]), float(epsilon), rel_tol=1e-9, abs_tol=0.0):
            warnings.warn(
                f"{key}: requested epsilon={float(epsilon):g} is not on the curve grid; "
                f"using nearest available epsilon={float(row['epsilon']):g}",
                stacklevel=2,
            )
        row["audited_epsilon"] = float(row["epsilon"])
        row["requested_epsilon"] = float(epsilon)
        rows.append(row)
    return pd.DataFrame(rows)


def add_audit_metadata(points: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return points.copy()
    keys = ["dataset", "architecture", "center_id"]
    keep = [
        "dataset",
        "architecture",
        "center_id",
        "audit_route",
        "bounded_domain",
        "certified_statement",
    ]
    merged = points.merge(audit[[c for c in keep if c in audit.columns]], on=keys, how="left")
    merged["audited_radius_source"] = "declared score-radius grid point from Figure 2 relaxed-margin audit"
    merged["claim"] = (
        "certifies every attacker whose normalized scores stay within audited_epsilon "
        "of this frozen center on the declared bounded trace domain"
    )
    return merged


def selected_epsilons(df: pd.DataFrame) -> list[float]:
    return sorted({float(x) for x in df["audited_epsilon"]}) if "audited_epsilon" in df else []


def make_tex_table(df: pd.DataFrame, epsilon: float) -> str:
    eps_tex = ", ".join(fmt_eps(e) for e in (selected_epsilons(df) or [float(epsilon)]))
    rows = []
    for _, r in sort_points(df).iterrows():
        rows.append(
            " & ".join(
                [
                    esc(DATASET_LABELS.get(str(r["dataset"]), str(r["dataset"]))),
                    esc(str(r["architecture"])),
                    rf"${fmt_eps(r['audited_epsilon'])}$",
                    fmt_float(r["A"], 3) if pd.notna(r.get("A")) else "--",
                    str(int(r["S_loc"])),
                    fmt_float(r["p_hat_loc"], 5),
                    fmt_float(r["p_loc_plus"], 5),
                    fmt_bits(r["BI_loc_plus"]),
                    esc(str(r["score_normalization"])),
                ]
            )
            + r" \\"
        )
    body = "\n    ".join(rows)
    return rf"""\begin{{table}}[t]
    \centering
    \caption{{Concrete audited score-radius points from the Figure~\ref{{fig:biloc-curve}} source data at $\varepsilon={eps_tex}$. These rows instantiate the local-family certificate for one MLP, one CNN, and one Transformer center on the same benchmark; each endpoint is the KL-binomial relaxed-margin $\mathrm{{BI}}^{{\mathrm{{loc}},+}}$ value at the audited radius.}}
    \label{{tab:audited-score-radius-biloc}}
    \scriptsize
    \setlength{{\tabcolsep}}{{3pt}}
    \resizebox{{\linewidth}}{{!}}{{%
    \begin{{tabular}}{{llrrrrrrl}}
    \toprule
    Benchmark & Arch. & $\varepsilon_{{\mathrm{{aud}}}}$ & $A$ & $S^{{\mathrm{{loc}}}}$ & $\widehat p^{{\mathrm{{loc}}}}$ & $p^{{\mathrm{{loc}},+}}$ & $\mathrm{{BI}}^{{\mathrm{{loc}},+}}$ & Normalizer \\
    \midrule
    {body}
    \bottomrule
    \end{{tabular}}}}
\end{{table}}
"""


def write_summary(path: Path, all_points: pd.DataFrame, main_points: pd.DataFrame, epsilon: float, main_dataset: str) -> None:
    selected = selected_epsilons(all_points) or [float(epsilon)]
    radius_line = f"Audited radius: `epsilon={', '.join(f'{e:g}' for e in selected)}`."
    if any(not math.isclose(e, float(epsilon), rel_tol=1e-9, abs_tol=0.0) for e in selected):
        radius_line += f" Requested `epsilon={epsilon:g}` is not on the curve grid; the nearest grid value was used."
    lines = [
        "# Audited Score-Radius BI_loc Points",
        "",
        radius_line,
        f"Main benchmark: `{main_dataset}`.",
        "",
        "These rows are concrete points on the Figure 2 score-radius sweep. They report the KL-binomial relaxed-margin endpoint at the declared audited score radius, not a parameter-space architecture radius.",
        "",
        "## Main rows",
        "",
        "| dataset | architecture | epsilon_aud | A | S_loc | p_hat_loc | p_loc_plus | BI_loc_plus |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in sort_points(main_points).iterrows():
        lines.append(
            f"| {r['dataset']} | {r['architecture']} | {float(r['audited_epsilon']):.6g} | "
            f"{float(r['A']):.6g} | {int(r['S_loc'])} | {float(r['p_hat_loc']):.6f} | "
            f"{float(r['p_loc_plus']):.6f} | {float(r['BI_loc_plus']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- All audited-point rows: `{len(all_points)}`",
            f"- Main benchmark rows: `{len(main_points)}`",
            "- Source curve: `bi_loc_radius_curves.csv`",
            "- Figure overlay: star markers on `figure2_biloc_all_datasets_selected_center_curves.pdf`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def update_manifest(package_dir: Path, entries: list[dict[str, str]]) -> None:
    manifest_path = package_dir / "artifact_manifest.csv"
    if not manifest_path.exists():
        return
    manifest = pd.read_csv(manifest_path)
    column_map = {col.lower(): col for col in manifest.columns}
    slot_col = column_map.get("slot", "slot")
    normalized_entries = []
    for entry in entries:
        normalized_entries.append(
            {
                column_map.get("slot", "slot"): entry["slot"],
                column_map.get("artifact", "artifact"): entry["artifact"],
                column_map.get("purpose", "purpose"): entry["purpose"],
            }
        )
    manifest = manifest[~manifest[slot_col].isin({entry["slot"] for entry in entries})]
    manifest = pd.concat([manifest, pd.DataFrame(normalized_entries)], ignore_index=True)
    manifest.to_csv(manifest_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves-csv", type=Path, default=Path("results/bi_scope_certificates/bi_loc_radius_curves.csv"))
    parser.add_argument("--audit-csv", type=Path, default=Path("results/bi_scope_certificates/nonoptional_audit/score_radius_audit_table.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/bi_scope_certificates"))
    parser.add_argument("--package-dir", type=Path, default=Path("results/bi_scope_certificates/mainpaper_package"))
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--main-dataset", default="ascad_desync_50")
    args = parser.parse_args()

    curves = pd.read_csv(args.curves_csv)
    audit = pd.read_csv(args.audit_csv) if args.audit_csv.exists() else pd.DataFrame()
    all_points = sort_points(add_audit_metadata(nearest_epsilon_rows(curves, args.epsilon), audit))
    main_points = all_points[all_points["dataset"] == args.main_dataset].copy()
    if main_points.empty:
        raise SystemExit(f"no audited points found for main dataset {args.main_dataset!r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / "audited_score_radius_biloc_points.csv"
    main_path = args.output_dir / "audited_score_radius_biloc_points_main.csv"
    all_points.to_csv(all_path, index=False)
    main_points.to_csv(main_path, index=False)
    write_summary(args.output_dir / "AUDITED_SCORE_RADIUS_BILOC_POINTS.md", all_points, main_points, args.epsilon, args.main_dataset)

    if args.package_dir:
        (args.package_dir / "data").mkdir(parents=True, exist_ok=True)
        (args.package_dir / "tables").mkdir(parents=True, exist_ok=True)
        all_points.to_csv(args.package_dir / "data" / "audited_score_radius_biloc_points.csv", index=False)
        main_points.to_csv(args.package_dir / "data" / "audited_score_radius_biloc_points_main.csv", index=False)
        (args.package_dir / "tables" / "audited_score_radius_biloc_points.tex").write_text(
            make_tex_table(main_points, args.epsilon),
            encoding="utf-8",
        )
        update_manifest(
            args.package_dir,
            [
                {
                    "slot": "Audited score-radius points",
                    "artifact": "data/audited_score_radius_biloc_points.csv",
                    "purpose": "Concrete epsilon="
                    + ", ".join(f"{e:g}" for e in (selected_epsilons(all_points) or [float(args.epsilon)]))
                    + " BI_loc points for Figure 2",
                },
                {
                    "slot": "Audited score-radius main table",
                    "artifact": "tables/audited_score_radius_biloc_points.tex",
                    "purpose": "Main-benchmark MLP/CNN/Transformer audited-radius table",
                },
            ],
        )

    print(all_path)
    print(main_path)


if __name__ == "__main__":
    main()
