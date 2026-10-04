# scripts

**What this folder does.** Table builders, summaries and the plot code for the diagnostic figures of the full version. The two figures of the camera-ready paper do not depend on it: they are drawn by `paper_results/figure1_dimension_sweep.py` and `paper_results/figure2_biloc_selected_centers.py` (the latter with `plot_bi_loc_radius_curves.py`).

## Files

| File | Purpose |
|---|---|
| `generate_paper_plots.py` | regenerates the diagnostic plots `plot_A_*.pdf` to `plot_G_*.pdf` from the result CSVs |
| `plot_bi_loc_radius_curves.py` | draws `BI_loc` radius curves (`bi_loc_radius_curves.pdf/.png` by default; `--output-stem`, `--point-label`) |
| `build_bi_suite_paper_artifacts.py` | finite-suite BI tables from seed rows and baseline summaries |
| `audit_bi_suite_margin_reduction.py` | audits margin-reduction rows used in suite tightness diagnostics |
| `build_full_suite_bi_loc_artifacts.py` | full-suite score-local BI tables |
| `build_audited_score_radius_points.py` | audited score-radius points |
| `build_real_challenger_diagnostic.py` | summarizes the real-data challenger diagnostics; writes `challenger_table.tex` (`tab:bayes-completeness-real`) |
| `merge_ascad_fixed_split_mlp_challengers.py` | merges ASCAD fixed-split MLP challenger rows |

## Run

```bash
python -m scripts.generate_paper_plots --out-dir results/paper_plots/Figures
python -m scripts.plot_bi_loc_radius_curves --help
```

`generate_paper_plots.py` draws only measured CHES non-BI cells in `plot_B_stability_vs_dimension.pdf`. `--fill-missing-dimension-cells` fills missing cells with hard-coded expected or log-interpolated values that are not measurements; it prints a warning per filled cell and draws them with hollow markers. `plot_E_bi_vs_holdout.pdf` skips a dataset, with a warning, when its summary is missing or its `p_obs`, `M` or `n_te` is not finite. The fixed-split overrides read `ascad_desync_0_attack_summary.csv` and `dpav4_attack_summary.csv` from `--fixed-split-summary-dir`, written by `bi_suite_8datasets/evaluate_tches20_pretrained_fixed_split.py --split attack`.

## Inputs with a producer in this repository

| Input of `generate_paper_plots.py` | Produced by |
|---|---|
| `aesrd_budget_checkpoint.csv` | `attacker_scope_diagnostics/aesrd_scope_seed_runner.py` |
| `scope_checkpoint.csv` | `attacker_scope_diagnostics/` |
| `tau_sweep_results.csv` | `locality_diagnostics/ascad_temporal_window_sweep.py` |
| dimension-sweep summaries | `dimension_stability/` |
| CHES Transformer and ASCAD EstraNet summaries | `bi_suite_8datasets/summarize_*.py` |

## Inputs without a producer in this repository

- `results/paper_artifacts/real_nonbi_dimcap_combined_summary.csv` (`--nonbi-summary`).
- `bi_suite_paper_table.csv` and `bi_suite_seed_rows.csv` come from `build_bi_suite_paper_artifacts.py`, which reads `results/reusable_suite_rows/reusable_suite_success.csv` and, if present, `results/baseline_metrics/baseline_checkpoint.csv`; neither is produced here.
- `build_real_challenger_diagnostic.py` reads `ascad_biarch_measured_tradeoff_points.csv`, `ches_ctf_anchor_frontier.csv` and `ascad_random_anchor_frontier.csv`.
- `build_audited_score_radius_points.py` reads `score_radius_audit_table.csv` (optional; audit metadata is omitted when missing).
- `audit_bi_suite_margin_reduction.py` reads `table1_bi_suite_eight_benchmarks.csv` (the BI-suite table, Table 6 in the paper) and `reusable_suite_success.csv`.
- The full-version figure `figure4_architecture_scope_diagnostic.pdf` is not written by these scripts.

## Notes on the challenger scripts

- `merge_ascad_fixed_split_mlp_challengers.py` defaults to `results/ascad_fixed_split_challenger`, the output directory of `attacker_scope_diagnostics/evaluate_ascad_d0_challenger_fixed_split.py`, and expects one evaluator run per seed in `<base-dir>/seed_<k>/`. Its default `--suite-p-plus` is the ASCAD d0 profiling-split endpoint, not the paper's attack-split endpoint, and its `p_fixed` is in-sample (scored on the full profiling set, which includes the challengers' training traces).
- `build_real_challenger_diagnostic.py` computes its action and interpretation text from the input CSVs.

Write all generated tables and plots under `results/` or another ignored path.
