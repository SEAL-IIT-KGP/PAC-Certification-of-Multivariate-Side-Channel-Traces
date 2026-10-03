# Experiment Map

This file maps the paper's experimental results to code entry points. Table and
figure numbers follow the camera-ready ASIACRYPT 2026 version (Tables 1-9,
Figures 1-2); the LaTeX label is given in parentheses. Results that appear only
in the full version of the paper are listed by label in the second part.

`paper_results/` regenerates every result of the first part and compares it
with the printed values; see `paper_results/README.md`.

## Camera-ready results

### Table 7 (`tab:bi-suite-eight-datasets`): finite-suite certificates and EPI

- Per-model attack success counts:
  `bi_suite_8datasets/evaluate_tches20_pretrained_fixed_split.py` (six TCHES20
  rows, `--split attack`) and `bi_suite_8datasets/evaluate_estranet_bi_suite.py`
  (ASCAD random-key and CHES-CTF-2025 rows), driven by
  `paper_results/reproduce.py tches20` and `paper_results/evaluate_estranet_suites.py`.
- Certificate arithmetic: `summarize_counts` in the TCHES20 evaluator, also
  available as `core/bi_certificate.py`.
- EPI column: `paper_results/evaluate_selected_models.py` and
  `paper_results/evaluate_estranet_suites.py` (helpers in `paper_results/matched_metrics.py`).
- Table assembly and check: `paper_results/table7_bi_suite.py`.

### Table 8 (`tab:matched-key-recovery`): key-byte recovery of the selected models

- `paper_results/evaluate_selected_models.py` (TCHES20 benchmarks) and
  `paper_results/evaluate_estranet_suites.py` (ASCAD random key).
- Table assembly and check: `paper_results/table8_key_recovery.py`.

### Figure 1 (`fig:metrics-vs-dim`): CHES-CTF-2025 dimension sweep

- `BI_suite,+` line: `bi_suite_8datasets/evaluate_estranet_bi_suite.py` on the
  window suites (`paper_results/evaluate_estranet_suites.py --suites ches_windows`);
  `bi_suite_8datasets/summarize_ches_transformer_bi_sweep.py` summarizes the same counts.
- GKOV MI, MLP-PI, ePI/PI cells: `dimension_stability/ches_nonbi_metric_cell.py`
  and `dimension_stability/real_nonbi_dimcap_baselines.py`.
- Figure and check: `paper_results/figure1_dimension_sweep.py`.

### Figure 2 (`fig:biloc-curve`): selected-center local-family curves and calibrated radius

- Relaxed-margin counts with the 20%/80% calibration split:
  `bi_scope_certificates/profile_epsilon_calibration.py`.
- Calibration rule, certificates, figure, and check:
  `paper_results/figure2_biloc_selected_centers.py` (plot style from
  `scripts/plot_bi_loc_radius_curves.py`).
- Bounds quoted in Section 5.3 at `eps = 1e-4` (full attack sets):
  `bi_scope_certificates/compute_bi_loc_margin_curves.py`.
- AES-RD input file of the Transformer reference model:
  `bi_scope_certificates/prepare_aesrd_estranet_h5.py`.

### Table 9 (`tab:biloc-fullsuite`): full-suite local-family certificate

- `bi_scope_certificates/compute_full_suite_bi_loc.py`.
- Table assembly and check: `paper_results/table9_biloc_full_suite.py`.

Tables 1-6 and the pipeline illustration are descriptive and have no producing script.

## Shared utilities

- `core/bi_certificate.py`, `core/baseline_metrics.py`, `core/data_loader.py`:
  BI endpoint helpers, PI/HI/MI baselines, dataset loaders.
- `datasets/README.md`, `datasets/setup_datasets.py`: dataset layout and setup helpers.
- `model_training/`: illustrative MLP, CNN, and Transformer-style training
  examples. The paper's suites are the pretrained TCHES20 models and the
  EstraNet checkpoints, not models produced by these examples.

## Results of the full version

### Attack-set size, comparison with density-style estimators, suite growth (`fig:bi-vs-holdout`, `fig:bi-vs-nonbi`, `fig:suite-growth`)

- `scripts/generate_paper_plots.py` writes `plot_E_bi_vs_holdout.pdf`,
  `plot_A_bi_vs_metrics.pdf`, and `plot_G_suite_growth.pdf`.
- Supporting tables: `scripts/build_bi_suite_paper_artifacts.py`,
  `scripts/audit_bi_suite_margin_reduction.py`.

### Temporal-window sweep (`fig:tau-sweep`)

- Data: `locality_diagnostics/ascad_temporal_window_sweep.py`; plot
  (`plot_C_tau_sweep.pdf`): `scripts/generate_paper_plots.py`.

### Reduced-dimension diagnostics

- `dimension_stability/real_nonbi_dimcap_baselines.py`,
  `dimension_stability/real_projected_bi_dimcap.py`,
  `dimension_stability/multivariate_stability_driver.py`;
  `scripts/generate_paper_plots.py` writes `plot_B_stability_vs_dimension.pdf`.

### Trained-scope and challenger diagnostics (`fig:scope-comparison`, `tab:bayes-completeness-real`)

- `attacker_scope_diagnostics/attacker_scope_diagnostic.py`,
  `attacker_scope_diagnostics/scope_cell_runner.py`,
  `attacker_scope_diagnostics/aesrd_scope_seed_runner.py`,
  `attacker_scope_diagnostics/evaluate_ascad_d0_challenger_fixed_split.py`.
- `scripts/build_real_challenger_diagnostic.py`,
  `scripts/merge_ascad_fixed_split_mlp_challengers.py`.

### Known-posterior oracle diagnostics (`tab:bayes-completeness-synthetic`)

- `bayes_oracle/gaussian_oracle_suite.py` (byte-scale Gaussian oracle row;
  run and checked by `paper_results/reproduce.py archived`).
- `bayes_oracle/bayes_oracle_synthetic.py` (masked locality oracle row).

### Radius calibration counts (`tab:radius-calibration`)

- `bi_scope_certificates/profile_epsilon_calibration.py` and
  `paper_results/figure2_biloc_selected_centers.py`.

### Additional local-family tables

- `scripts/build_full_suite_bi_loc_artifacts.py`,
  `scripts/build_audited_score_radius_points.py`,
  `bi_scope_certificates/build_bi_scope_tables.py`.

### Four-byte key recovery on raw ASCAD traces (`tab:raw-ascad-four-byte-composition`, quoted in Section 5.1)

- `multibyte_composition/fetch_raw_ascad.py`, `scan_raw_ascad_pois.py`,
  `train_raw_ascad_byte_attacker.py`, `compose_four_byte_recovery.py`;
  check in `paper_results/fullversion_tables.py` (`paper_results.reproduce fullversion`).

### PI/EPI under temperature rescaling (`tab:epi-temperature`, summarized in Section 5.2)

- `epi_comparison/epi_head_to_head.py` and `summarize_epi_head_to_head.py`;
  check in `paper_results/fullversion_tables.py`.

### Figure files drawn outside this repository

- `figure4_architecture_scope_diagnostic.pdf` (`fig:scope-comparison`) is
  drawn from the outputs of `attacker_scope_diagnostics/`.
