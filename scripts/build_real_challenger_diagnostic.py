#!/usr/bin/env python3
"""Build the real-data out-of-family challenger diagnostic artifacts.

This is a synthesis step over completed experiment outputs.  It intentionally
keeps known-posterior oracle checks separate from real-data challenger
diagnostics, because the latter cannot certify unrestricted Bayes optimality.
"""

from __future__ import annotations

import math
import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


K = 256
DEFAULT_OUT_DIR = Path("results/real_challenger_diagnostic")


def bi_from_p(p: float, k: int = K) -> float:
    """Advantage-normalized BI used in the paper tables."""
    if not math.isfinite(p):
        return float("nan")
    return max(0.0, min(math.log2(k), math.log2(1.0 + k * max(p - 1.0 / k, 0.0))))


def fmt(x: float | int | str | None, digits: int = 6) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        if pd.isna(x):
            return ""
    except TypeError:
        pass
    if isinstance(x, int):
        return str(x)
    return f"{float(x):.{digits}f}"


@dataclass(frozen=True)
class SourcePaths:
    suite: Path = Path("results/bi_suite_8datasets/bi_suite_paper_table.csv")
    ascad_arch: Path = Path("results/ascad_arch_tradeoff/ascad_biarch_measured_tradeoff_points.csv")
    ascad_scope: Path = Path("results/attacker_scope_diagnostic/scope_checkpoint.csv")
    aes_budget: Path = Path("results/attacker_scope_diagnostic/aesrd_budget_checkpoint.csv")
    ches_width: Path = Path("results/ches_transformer_bi_dim_sweep/ches_transformer_bi_dim_sweep_summary.csv")
    ches_anchor: Path = Path("results/ches_ctf_radius_anchor_sweep/ches_ctf_anchor_frontier.csv")
    ascad_random_anchor: Path = Path("results/ascad_random_radius_anchor_sweep/ascad_random_anchor_frontier.csv")


def load_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def suite_lookup(suite: pd.DataFrame, dataset: str) -> dict[str, object]:
    row = suite[suite["dataset"] == dataset]
    if row.empty:
        raise KeyError(f"missing suite row for {dataset}")
    r = row.iloc[0].to_dict()
    return {
        "original_scope": r["scope_id"],
        "suite_M": int(r["M"]),
        "suite_n_te": int(r["n_te"]),
        "suite_p_obs": float(r["p_obs"]),
        "suite_p_plus": float(r["p_plus"]),
        "suite_BI_obs": float(r["BI_obs"]),
        "suite_BI_plus": float(r["BI_plus"]),
        "suite_best_model": r["best_model"],
    }


def make_row(
    suite: pd.DataFrame,
    dataset: str,
    challenger_family: str,
    challenger: str,
    challenger_p_hat: float,
    challenger_BI_obs: float | None,
    source_artifact: str,
    n_te: int | None = None,
    challenger_p_plus: float | None = None,
    challenger_BI_plus: float | None = None,
    challenger_M: int | None = None,
    evaluation_mode: str = "diagnostic",
    comparability: str = "same/frozen attack split or augmented-scope diagnostic",
    action: str = "No redeclaration needed.",
) -> dict[str, object]:
    base = suite_lookup(suite, dataset)
    p = float(challenger_p_hat)
    bi_obs = bi_from_p(p) if challenger_BI_obs is None else float(challenger_BI_obs)
    p_plus = None if challenger_p_plus is None else float(challenger_p_plus)
    bi_plus = None if challenger_BI_plus is None else float(challenger_BI_plus)
    exceeds_suite_p_obs = p > float(base["suite_p_obs"])
    exceeds_suite_p_plus = p > float(base["suite_p_plus"])
    material_delta = p - float(base["suite_p_obs"])
    row = {
        "dataset": dataset,
        **base,
        "challenger_family": challenger_family,
        "challenger": challenger,
        "challenger_M": challenger_M,
        "challenger_n_te": n_te,
        "challenger_p_hat": p,
        "challenger_p_plus": p_plus,
        "challenger_BI_obs": bi_obs,
        "challenger_BI_plus": bi_plus,
        "delta_vs_suite_p_obs": material_delta,
        "exceeds_suite_p_obs": bool(exceeds_suite_p_obs),
        "exceeds_suite_p_plus": bool(exceeds_suite_p_plus),
        "evaluation_mode": evaluation_mode,
        "comparability": comparability,
        "action": action,
        "source_artifact": source_artifact,
    }
    return row


def comparison_action(p: float, base: dict[str, object], subject: str, reference: str) -> str:
    """Action text derived from the challenger-versus-suite comparison."""
    if p > float(base["suite_p_plus"]):
        return f"{subject} exceeds {reference} upper certificate p_plus; redeclare the scope before using this row."
    if p > float(base["suite_p_obs"]):
        return f"{subject} exceeds {reference} point estimate but remains within p_plus."
    return f"{subject} does not exceed {reference} point estimate."


def build_rows(paths: SourcePaths) -> pd.DataFrame:
    suite = load_required(paths.suite)
    rows: list[dict[str, object]] = []

    arch = load_required(paths.ascad_arch)
    for (category, category_label), g in arch.groupby(["category", "category_label"], dropna=False):
        best = g.sort_values("attack_success", ascending=False).iloc[0]
        arch_n_te = next((int(best[col]) for col in ("n_te", "n_holdout") if col in best.index and pd.notna(best[col])), None)
        rows.append(
            make_row(
                suite,
                dataset="ascad_desync_0",
                challenger_family=f"ASCAD architecture challengers: {category_label}",
                challenger=f"{best['lane_label']} / {best['tag']}",
                challenger_M=len(g),
                challenger_p_hat=float(best["attack_success"]),
                challenger_BI_obs=float(best["observed_bi_bits"]),
                source_artifact=str(paths.ascad_arch),
                n_te=arch_n_te,
                evaluation_mode="frozen measured challenger group",
                comparability="same ASCAD target family; reported as a measured challenger group",
                action=comparison_action(
                    float(best["attack_success"]),
                    suite_lookup(suite, "ascad_desync_0"),
                    "Best challenger in this group",
                    "the frozen suite",
                ),
            )
        )

    scope = load_required(paths.ascad_scope)
    scope = scope[scope["dataset"] == "ascad_desync_0"]
    for scope_name, g in scope.groupby("scope_name", dropna=False):
        best = g.sort_values("p_max", ascending=False).iloc[0]
        p = float(best["p_max"])
        scope_suite = suite_lookup(suite, "ascad_desync_0")
        suite_plus = scope_suite["suite_p_plus"]
        n_holdout = int(best["n_holdout"])
        rows.append(
            make_row(
                suite,
                dataset="ascad_desync_0",
                challenger_family="ASCAD augmented scope challenger",
                challenger=f"{scope_name}; best={best['best_model']}; seed={int(best['seed'])}",
                challenger_M=int(best["M"]),
                challenger_p_hat=p,
                challenger_p_plus=None,
                challenger_BI_obs=None,
                challenger_BI_plus=float(best["bi"]),
                source_artifact=str(paths.ascad_scope),
                n_te=n_holdout,
                evaluation_mode=f"separate {n_holdout}-trace holdout / augmented-scope diagnostic",
                comparability=(
                    f"not the frozen suite split (suite n_te={scope_suite['suite_n_te']}); "
                    "use only as redeclaration diagnostic"
                ),
                action=(
                    "Point estimate exceeds suite p_plus; redeclaration pressure: redeclare an augmented scope before this row is used."
                    if p > suite_plus
                    else "Point estimate exceeds suite p_obs but remains below suite p_plus; redeclare an augmented scope if this row is used."
                    if p > scope_suite["suite_p_obs"]
                    else "No redeclaration pressure from this diagnostic row."
                ),
            )
        )

    aes = load_required(paths.aes_budget)
    for scope_name, g in aes.groupby("scope_name", dropna=False):
        best = g.sort_values("p_max", ascending=False).iloc[0]
        rows.append(
            make_row(
                suite,
                dataset="aes_rd",
                challenger_family="AES-RD budget/scope challenger",
                challenger=f"{scope_name}; best={best['best_model']}; seed={int(best['seed'])}",
                challenger_M=int(best["M"]),
                challenger_p_hat=float(best["p_max"]),
                challenger_BI_obs=None,
                challenger_BI_plus=float(best["bi"]),
                source_artifact=str(paths.aes_budget),
                n_te=int(best["n_holdout"]),
                evaluation_mode="budgeted challenger holdout",
                comparability="diagnostic budget/scope split; separate from the frozen AES-RD suite",
                action=comparison_action(
                    float(best["p_max"]),
                    suite_lookup(suite, "aes_rd"),
                    "Best budget/scope challenger",
                    "the frozen AES-RD suite",
                ),
            )
        )

    ches_width = load_required(paths.ches_width)
    for _, r in ches_width.sort_values("p_obs", ascending=False).iterrows():
        dim = int(r["dimension"])
        rows.append(
            make_row(
                suite,
                dataset="ches_ctf_2025",
                challenger_family="CHES Transformer width challenger",
                challenger=f"EstraNet window/width {dim}; best={r['best_model']}",
                challenger_M=int(r["M"]),
                challenger_p_hat=float(r["p_obs"]),
                challenger_p_plus=float(r["p_plus"]),
                challenger_BI_obs=float(r["BI_obs"]),
                challenger_BI_plus=float(r["BI_plus"]),
                source_artifact=str(paths.ches_width),
                n_te=int(r["n_te"]),
                evaluation_mode="width sweep certificate",
                comparability="same 100k CHES holdout; alternative Transformer window suite",
                action=(
                    "Higher point estimate than w3000 and above original p_plus; redeclare the width-sweep suite before this row is reported."
                    if float(r["p_obs"]) > suite_lookup(suite, "ches_ctf_2025")["suite_p_plus"]
                    else "Higher point estimate than w3000, but still inside original p_plus; redeclare width-sweep suite only if this becomes the reported scope."
                    if float(r["p_obs"]) > suite_lookup(suite, "ches_ctf_2025")["suite_p_obs"]
                    else "No point-estimate improvement over the frozen w3000 suite."
                ),
            )
        )

    ches_anchor = load_required(paths.ches_anchor)
    for _, r in ches_anchor.sort_values("success", ascending=False).iterrows():
        rows.append(
            make_row(
                suite,
                dataset="ches_ctf_2025",
                challenger_family="CHES checkpoint anchor challenger",
                challenger=str(r["anchor"]),
                challenger_M=1,
                challenger_p_hat=float(r["success"]),
                challenger_BI_obs=float(r["observed_bi_bits"]),
                challenger_BI_plus=float(r["finite_kl_bi_plus_bits"]),
                source_artifact=str(paths.ches_anchor),
                n_te=100000,
                evaluation_mode="checkpoint anchor sweep",
                comparability="same CHES holdout; individual anchor relative to best checkpoint",
                action=comparison_action(
                    float(r["success"]),
                    suite_lookup(suite, "ches_ctf_2025"),
                    "This anchor",
                    "the frozen checkpoint-suite",
                ),
            )
        )

    ascad_random_anchor = load_required(paths.ascad_random_anchor)
    for _, r in ascad_random_anchor.sort_values("success", ascending=False).iterrows():
        rows.append(
            make_row(
                suite,
                dataset="ascad_random_key",
                challenger_family="ASCAD-random checkpoint anchor challenger",
                challenger=str(r["anchor"]),
                challenger_M=1,
                challenger_p_hat=float(r["success"]),
                challenger_BI_obs=float(r["observed_bi_bits"]),
                challenger_BI_plus=float(r["finite_kl_bi_plus_bits"]),
                source_artifact=str(paths.ascad_random_anchor),
                n_te=100000,
                evaluation_mode="checkpoint anchor sweep",
                comparability="same 100k ASCAD-random holdout; individual anchor relative to best checkpoint",
                action=comparison_action(
                    float(r["success"]),
                    suite_lookup(suite, "ascad_random_key"),
                    "This anchor",
                    "the frozen checkpoint-suite",
                ),
            )
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["dataset", "challenger_family", "challenger_p_hat"], ascending=[True, True, False])
    return out.reset_index(drop=True)


def build_dataset_summary(rows: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for dataset, g in rows.groupby("dataset"):
        best = g.sort_values("challenger_p_hat", ascending=False).iloc[0]
        pieces.append(
            {
                "dataset": dataset,
                "original_scope": best["original_scope"],
                "suite_p_obs": best["suite_p_obs"],
                "suite_p_plus": best["suite_p_plus"],
                "best_challenger": best["challenger"],
                "best_challenger_family": best["challenger_family"],
                "best_challenger_p_hat": best["challenger_p_hat"],
                "best_delta_vs_suite_p_obs": best["delta_vs_suite_p_obs"],
                "best_exceeds_suite_p_obs": best["exceeds_suite_p_obs"],
                "best_exceeds_suite_p_plus": best["exceeds_suite_p_plus"],
                "action": best["action"],
            }
        )
    return pd.DataFrame(pieces).sort_values("dataset").reset_index(drop=True)


def write_markdown(rows: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    table_rows = []
    for _, r in summary.iterrows():
        table_rows.append(
            "| {dataset} | {suite_p_obs} | {suite_p_plus} | {challenger} | {p_hat} | {delta} | {exceeds_plus} | {action} |".format(
                dataset=r["dataset"],
                suite_p_obs=fmt(r["suite_p_obs"]),
                suite_p_plus=fmt(r["suite_p_plus"]),
                challenger=r["best_challenger"],
                p_hat=fmt(r["best_challenger_p_hat"]),
                delta=fmt(r["best_delta_vs_suite_p_obs"], digits=6),
                exceeds_plus="yes" if r["best_exceeds_suite_p_plus"] else "no",
                action=r["action"],
            )
        )

    point_exceed = int(rows["exceeds_suite_p_obs"].sum())
    plus_exceed = int(rows["exceeds_suite_p_plus"].sum())
    increase_families = sorted(rows.loc[rows["exceeds_suite_p_obs"].astype(bool), "challenger_family"].astype(str).unique())
    plus_families = sorted(rows.loc[rows["exceeds_suite_p_plus"].astype(bool), "challenger_family"].astype(str).unique())
    if plus_exceed:
        increase_claim = (
            f"- Rows exceeding the frozen suite upper certificate `p_plus` come from: {'; '.join(plus_families)}. "
            "These require scope redeclaration before they are reported."
        )
    elif point_exceed:
        increase_claim = (
            f"- The only point-estimate increases come from: {'; '.join(increase_families)}; "
            "none exceeds the frozen suite upper certificate `p_plus`."
        )
    else:
        increase_claim = "- No row exceeds the frozen suite point estimate."

    def best_row(dataset: str, family_prefix: str) -> pd.Series | None:
        g = rows[(rows["dataset"] == dataset) & rows["challenger_family"].astype(str).str.startswith(family_prefix)]
        return None if g.empty else g.sort_values("challenger_p_hat", ascending=False).iloc[0]

    interpretation = []
    ascad_parts = []
    arch_best = best_row("ascad_desync_0", "ASCAD architecture challengers")
    if arch_best is not None:
        ascad_parts.append(
            f"measured architecture challengers {'exceed' if arch_best['exceeds_suite_p_obs'] else 'do not exceed'} "
            f"the frozen TCHES20 suite point estimate (best {fmt(arch_best['challenger_p_hat'])} vs {fmt(arch_best['suite_p_obs'])})."
        )
    scope_best = best_row("ascad_desync_0", "ASCAD augmented scope challenger")
    if scope_best is not None:
        scope_n = f"{int(scope_best['challenger_n_te'])}-trace " if pd.notna(scope_best["challenger_n_te"]) else ""
        ascad_parts.append(
            f"A separate {scope_n}augmented-scope diagnostic reaches {fmt(scope_best['challenger_p_hat'])}, "
            f"{'above' if scope_best['exceeds_suite_p_plus'] else 'within'} the frozen suite upper bound "
            f"{fmt(scope_best['suite_p_plus'])} (suite n_te={int(scope_best['suite_n_te'])}); if used in the paper, "
            "it should be reported as an augmented-scope diagnostic rather than as the frozen-suite result."
        )
    if ascad_parts:
        interpretation.append("- ASCAD desync 0: " + " ".join(ascad_parts))
    aes_best = best_row("aes_rd", "")
    if aes_best is not None:
        interpretation.append(
            f"- AES-RD: budgeted linear/MLP/CNN challengers reach at most {fmt(aes_best['challenger_p_hat'])}, "
            f"{'above' if aes_best['exceeds_suite_p_obs'] else 'not above'} the frozen suite success {fmt(aes_best['suite_p_obs'])}."
        )
    random_best = best_row("ascad_random_key", "")
    if random_best is not None:
        interpretation.append(
            f"- ASCAD random key: checkpoint-anchor challengers {'beat' if random_best['exceeds_suite_p_obs'] else 'do not beat'} "
            f"the frozen EstraNet checkpoint-suite best ({fmt(random_best['challenger_p_hat'])} vs {fmt(random_best['suite_p_obs'])})."
        )
    width_best = best_row("ches_ctf_2025", "CHES Transformer width challenger")
    if width_best is not None:
        interpretation.append(
            f"- CHES-CTF-2025: the largest width-sweep point estimate is {fmt(width_best['challenger_p_hat'])} "
            f"({width_best['challenger']}) versus {fmt(width_best['suite_p_obs'])} for the reported w3000 suite; it is "
            f"{'above' if width_best['exceeds_suite_p_plus'] else 'within'} the reported suite upper certificate "
            f"{fmt(width_best['suite_p_plus'])}. The width sweep is best described as a saturation/challenger diagnostic "
            "or as a re-declared width-sweep suite if included."
        )
    body = [
        "# Real-Data Out-of-Family Challenger Diagnostic",
        "",
        "This artifact is a real-data challenger diagnostic only; it does not verify unrestricted oracle optimality because the real posterior is unknown.",
        "",
        "## Summary",
        "",
        f"- Challenger rows assembled: {len(rows)}.",
        f"- Rows whose point estimate exceeds the frozen suite point estimate: {point_exceed}.",
        f"- Rows whose point estimate exceeds the frozen suite upper certificate `p_plus`: {plus_exceed}.",
        increase_claim,
        "",
        "## Dataset-Level Best Challenger",
        "",
        "| Dataset | Suite p_obs | Suite p_plus | Best challenger | Challenger p_hat | Delta vs suite p_obs | Exceeds suite p_plus? | Action |",
        "|---|---:|---:|---|---:|---:|---|---|",
        *table_rows,
        "",
        "## Interpretation",
        "",
        *interpretation,
        "",
        "## Files",
        "",
        "- `challenger_rows.csv`: full row-level challenger table.",
        "- `challenger_dataset_summary.csv`: one best-challenger row per dataset.",
        "- `challenger_table.tex`: compact LaTeX table for an appendix.",
    ]
    out_path.write_text("\n".join(body) + "\n")


def latex_escape(text: object) -> str:
    s = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s


def write_latex(summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Out-of-family challenger diagnostics. These rows are diagnostics for declared-family saturation, not proofs of unrestricted oracle optimality.}",
        r"\label{tab:bayes-completeness-real}",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Dataset & $\widehat p_{\rm suite}$ & $p^+_{\rm suite}$ & $\widehat p_{\rm chall.}$ & $\Delta$ & Action \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        action = "No $p^+$ exceedance" if not r["best_exceeds_suite_p_plus"] else "Redeclare scope"
        lines.append(
            f"{latex_escape(r['dataset'])} & {float(r['suite_p_obs']):.5f} & {float(r['suite_p_plus']):.5f} & "
            f"{float(r['best_challenger_p_hat']):.5f} & {float(r['best_delta_vs_suite_p_obs']):+.5f} & {action} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    out_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--suite", type=Path, default=SourcePaths.suite)
    parser.add_argument("--ascad-arch", type=Path, default=SourcePaths.ascad_arch)
    parser.add_argument("--ascad-scope", type=Path, default=SourcePaths.ascad_scope)
    parser.add_argument("--aes-budget", type=Path, default=SourcePaths.aes_budget)
    parser.add_argument("--ches-width", type=Path, default=SourcePaths.ches_width)
    parser.add_argument("--ches-anchor", type=Path, default=SourcePaths.ches_anchor)
    parser.add_argument("--ascad-random-anchor", type=Path, default=SourcePaths.ascad_random_anchor)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(
        SourcePaths(
            suite=args.suite,
            ascad_arch=args.ascad_arch,
            ascad_scope=args.ascad_scope,
            aes_budget=args.aes_budget,
            ches_width=args.ches_width,
            ches_anchor=args.ches_anchor,
            ascad_random_anchor=args.ascad_random_anchor,
        )
    )
    summary = build_dataset_summary(rows)
    rows.to_csv(out_dir / "challenger_rows.csv", index=False)
    summary.to_csv(out_dir / "challenger_dataset_summary.csv", index=False)
    write_markdown(rows, summary, out_dir / "CHALLENGER_DIAGNOSTIC_SUMMARY.md")
    write_latex(summary, out_dir / "challenger_table.tex")
    print(f"wrote {len(rows)} challenger rows to {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
