# Postprocessing And Plot Scripts

This directory contains table builders, summary scripts, and paper-plot generation code.

- `build_bi_suite_paper_artifacts.py`: builds finite-suite BI paper tables from seed rows and baseline summaries.
- `audit_bi_suite_margin_reduction.py`: audits margin-reduction rows used in suite tightness diagnostics.
- `build_full_suite_bi_loc_artifacts.py`: builds full-suite score-local BI artifacts.
- `build_audited_score_radius_points.py`: prepares audited score-radius points.
- `plot_bi_loc_radius_curves.py`: generates BI_loc radius plots.
- `generate_paper_plots.py`: regenerates the main paper diagnostic plots from current CSV artifacts.
- `build_real_challenger_diagnostic.py`: summarizes real-data out-of-family challenger diagnostics.
- `merge_ascad_fixed_split_mlp_challengers.py`: merges ASCAD fixed-split MLP challenger rows.

Paper plots are generated with:

```bash
python -m scripts.generate_paper_plots \
  --out-dir results/paper_plots/Figures
```

`generate_paper_plots.py` writes `plot_A_*.pdf` to `plot_G_*.pdf` only. By default, `plot_B_stability_vs_dimension.pdf` shows only measured CHES non-BI (dimension, metric) cells. `--fill-missing-dimension-cells` fills missing cells with hard-coded expected or log-interpolated values that are not measurements. With this flag, the script prints a warning listing each filled (dimension, metric, status) and draws filled points with hollow markers. `plot_E_bi_vs_holdout.pdf` skips a dataset, with a warning, when its real summary is missing or its `p_obs`/`M`/`n_te` is not finite. ASCAD random is plotted only from the EstraNet summary. The fixed-split overrides read `ascad_desync_0_attack_summary.csv` and `dpav4_attack_summary.csv` from `--fixed-split-summary-dir`. These files are written by `bi_suite_8datasets/evaluate_tches20_pretrained_fixed_split.py --split attack`.

If you regenerate only local-family radius figures, use:

```bash
python -m scripts.plot_bi_loc_radius_curves --help
```

## Inputs not produced by any script in this repository

`generate_paper_plots.py` needs these inputs:

- `results/paper_artifacts/real_nonbi_dimcap_combined_summary.csv` (`--nonbi-summary`). No script writes this combined file.
- `bi_suite_paper_table.csv` and `bi_suite_seed_rows.csv` come from `build_bi_suite_paper_artifacts.py`. That script reads `results/reusable_suite_rows/reusable_suite_success.csv` and, if present, `results/baseline_metrics/baseline_checkpoint.csv`. Neither file is produced here.
- Its other inputs have producers:
  - `aesrd_budget_checkpoint.csv`: `attacker_scope_diagnostics/aesrd_scope_seed_runner.py`
  - `scope_checkpoint.csv`: `attacker_scope_diagnostics/`
  - `tau_sweep_results.csv`: `locality_diagnostics/ascad_temporal_window_sweep.py`
  - dimension-sweep summaries: `dimension_stability/`
  - CHES Transformer and ASCAD EstraNet summaries: `bi_suite_8datasets/summarize_*.py`

Other scripts here also read inputs that no script in this repository writes:

- `build_real_challenger_diagnostic.py`: `ascad_biarch_measured_tradeoff_points.csv`, `ches_ctf_anchor_frontier.csv`, and `ascad_random_anchor_frontier.csv`.
- `build_audited_score_radius_points.py`: `score_radius_audit_table.csv`. It is optional; audit metadata is omitted when the file is missing.
- `audit_bi_suite_margin_reduction.py`: `table1_bi_suite_eight_benchmarks.csv` (the BI-suite table, Table 6 in the paper) and `reusable_suite_success.csv`.

The paper figure files `figure1_ches_dimension_sweep.pdf`, `figure2_biloc_all_datasets_selected_center_curves.pdf`, and `figure4_architecture_scope_diagnostic.pdf` are not written by these scripts. `plot_bi_loc_radius_curves.py` writes `bi_loc_radius_curves.pdf`/`.png`.

## Challenger notes

- `merge_ascad_fixed_split_mlp_challengers.py` defaults to `results/ascad_fixed_split_challenger`, the same default output directory as `attacker_scope_diagnostics/evaluate_ascad_d0_challenger_fixed_split.py`. It expects one evaluator run per seed in `<base-dir>/seed_<k>/`.
  - Its default `--suite-p-plus` is the ASCAD d0 profiling-split endpoint, not the paper's attack-split endpoint.
  - Its `p_fixed` is in-sample: it is scored on the full profiling set, which includes the challengers' training traces.
- `build_real_challenger_diagnostic.py` writes `challenger_table.tex` with label `tab:bayes-completeness-real`. Its action and interpretation text is computed from the input CSVs.

All generated tables and plots should be written under `results/` or another ignored output path.
