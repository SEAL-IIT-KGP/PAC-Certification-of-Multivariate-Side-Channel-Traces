# scripts

Table builders, summaries and plot code for the diagnostic figures of the full version. The two figures of the camera-ready paper are drawn by `paper_results/` instead.

## Files

| File | Purpose |
|---|---|
| `generate_paper_plots.py` | the diagnostic plots `plot_A_*.pdf` to `plot_G_*.pdf` from the result CSVs |
| `plot_bi_loc_radius_curves.py` | `BI_loc` radius curves (also used by Figure 2) |
| `build_bi_suite_paper_artifacts.py` | finite-suite BI tables from seed rows and baseline summaries |
| `audit_bi_suite_margin_reduction.py` | margin-reduction audit for the suite diagnostics |
| `build_full_suite_bi_loc_artifacts.py` | full-suite `BI_loc` tables |
| `build_audited_score_radius_points.py` | audited score-radius points |
| `build_real_challenger_diagnostic.py` | the real-data challenger table (`challenger_table.tex`) |
| `merge_ascad_fixed_split_mlp_challengers.py` | merges ASCAD fixed-split MLP challenger rows |

## Run

```bash
python -m scripts.generate_paper_plots --out-dir results/paper_plots/Figures
python -m scripts.plot_bi_loc_radius_curves --help
```

## Inputs of `generate_paper_plots.py`

| Input | Produced by |
|---|---|
| `aesrd_budget_checkpoint.csv`, `scope_checkpoint.csv` | `attacker_scope_diagnostics/` |
| `tau_sweep_results.csv` | `locality_diagnostics/` |
| dimension-sweep summaries | `dimension_stability/` |
| CHES Transformer and ASCAD EstraNet summaries | `bi_suite_8datasets/summarize_*.py` |
| `ascad_desync_0_attack_summary.csv`, `dpav4_attack_summary.csv` (`--fixed-split-summary-dir`) | `bi_suite_8datasets/evaluate_tches20_pretrained_fixed_split.py --split attack` |

Write generated tables and plots under `results/`.
