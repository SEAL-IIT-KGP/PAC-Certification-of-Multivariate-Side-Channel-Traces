# Experiment Map

This file maps the manuscript experiment families to code entry points. Table and figure numbers follow the camera-ready ASIACRYPT 2026 version; the LaTeX label is given in parentheses.

## Shared utilities

- `core/bi_certificate.py`
- `core/baseline_metrics.py`
- `core/data_loader.py`
- `datasets/setup_datasets.py`

These files provide the shared BI endpoint helpers, PI/HI/MI baselines, dataset loaders, and optional dataset setup utilities used by the experiment entry points below.

## Datasets and model training

- `datasets/README.md`
- `datasets/setup_datasets.py`
- `model_training/common.py`
- `model_training/train_mlp_example.py`
- `model_training/train_cnn_example.py`
- `model_training/train_transformer_example.py`

These files document the eight benchmark rows and provide illustrative MLP, CNN, and Transformer-style training examples. The paper's suites are the pretrained TCHES20 and EstraNet reference models, not models produced by these examples.

## Main finite-suite certificates: Table 6 (`tab:bi-suite-eight-datasets`)

- `bi_suite_8datasets/evaluate_tches20_pretrained_fixed_split.py` (certify on the attack partition; see `bi_suite_8datasets/README.md`)
- `bi_suite_8datasets/evaluate_estranet_bi_suite.py`
- `bi_suite_8datasets/summarize_ascad_estranet_bi_baselines.py`
- `bi_suite_8datasets/summarize_ches_transformer_bi_sweep.py`
- `scripts/build_bi_suite_paper_artifacts.py`
- `scripts/audit_bi_suite_margin_reduction.py`

These scripts compute KL-binomial finite-suite BI brackets for the declared TCHES20-style and EstraNet checkpoint suites.

## High-dimensional stability: Figure 1 (`fig:metrics-vs-dim`)

- `dimension_stability/ches_nonbi_metric_cell.py`
- `dimension_stability/real_nonbi_dimcap_baselines.py`
- `dimension_stability/real_projected_bi_dimcap.py`
- `dimension_stability/multivariate_stability_driver.py`
- `bi_suite_8datasets/summarize_ches_transformer_bi_sweep.py`
- `scripts/generate_paper_plots.py` (writes `plot_B_stability_vs_dimension.pdf`)

These scripts run the dimension-sweep and reduced-dimension non-BI diagnostics. The paper's `figure1_ches_dimension_sweep.pdf` file itself is not written by a script in this repository.

## Local-family score-radius certificates: Figure 2 (`fig:biloc-curve`) and Table 7 (`tab:biloc-fullsuite`)

- `bi_scope_certificates/compute_bi_loc_margin_curves.py`
- `bi_scope_certificates/compute_full_suite_bi_loc.py`
- `bi_scope_certificates/build_bi_scope_tables.py`
- `scripts/plot_bi_loc_radius_curves.py`
- `scripts/build_full_suite_bi_loc_artifacts.py`
- `scripts/build_audited_score_radius_points.py`

These scripts compute and postprocess score-local `BI_loc` relaxed-margin certificates. The 20%/80% radius-calibration split of Table 12 (`tab:radius-calibration`) is not implemented here; the scripts certify on the attack partition they are given.

## Appendix plots: Figures 4–7

- Figure 4 (`fig:bi-vs-holdout`, `plot_E_bi_vs_holdout.pdf`), Figure 5 (`fig:bi-vs-nonbi`, `plot_A_bi_vs_metrics.pdf`), Figure 7 (`fig:suite-growth`, `plot_G_suite_growth.pdf`): `scripts/generate_paper_plots.py`.
- Figure 6 (`fig:tau-sweep`, `plot_C_tau_sweep.pdf`): data from `locality_diagnostics/ascad_temporal_window_sweep.py`, plotted by `scripts/generate_paper_plots.py`.

## Trained-scope and challenger diagnostics: Figure 8 (`fig:scope-comparison`) and Table 11 (`tab:bayes-completeness-real`)

- `attacker_scope_diagnostics/attacker_scope_diagnostic.py`
- `attacker_scope_diagnostics/scope_cell_runner.py`
- `attacker_scope_diagnostics/aesrd_scope_seed_runner.py`
- `attacker_scope_diagnostics/evaluate_ascad_d0_challenger_fixed_split.py`
- `scripts/build_real_challenger_diagnostic.py`
- `scripts/merge_ascad_fixed_split_mlp_challengers.py`

These scripts support the suite-size, scope, and challenger diagnostics. The paper's `figure4_architecture_scope_diagnostic.pdf` file itself is not written by a script in this repository.

## Known-posterior oracle diagnostics: Table 10 (`tab:bayes-completeness-synthetic`)

- `bayes_oracle/gaussian_oracle_suite.py` (byte-scale Gaussian oracle row)
- `bayes_oracle/bayes_oracle_synthetic.py` (masked locality oracle row)

## Not covered by this repository

- Figure 3 (`fig:bi-pipeline`) is an illustration.
- Table 9 (`tab:raw-ascad-four-byte-composition`) and Table 12 (`tab:radius-calibration`) have no producing script here.
