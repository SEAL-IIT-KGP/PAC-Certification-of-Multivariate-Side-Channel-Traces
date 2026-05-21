# Experiment Map

This file maps the manuscript experiment families to code entry points.

## Shared utilities

- `core/bi_certificate.py`
- `core/baseline_metrics.py`
- `core/data_loader.py`
- `datasets/setup_datasets.py`

These files provide the shared BI endpoint helpers, PI/HI/MI baselines, dataset loaders, and optional dataset setup utilities used by the experiment entry points below.

## Datasets and model training

- `datasets/README.md`
- `datasets/setup_datasets.py`
- `model_training/train_mlp_example.py`
- `model_training/train_cnn_example.py`
- `model_training/train_transformer_example.py`

These files document the eight benchmark rows and provide lightweight examples for training the MLP, CNN, and Transformer-style attackers used by the evaluation scripts.

## Main finite-suite certificates

- `bi_suite_8datasets/evaluate_tches20_pretrained_fixed_split.py`
- `bi_suite_8datasets/evaluate_estranet_bi_suite.py`
- `scripts/build_bi_suite_paper_artifacts.py`
- `scripts/audit_bi_suite_margin_reduction.py`

These scripts compute KL-binomial finite-suite BI brackets for the declared TCHES20-style and EstraNet checkpoint suites.

## High-dimensional stability and non-BI baselines

- `dimension_stability/multivariate_stability_driver.py`
- `dimension_stability/real_nonbi_dimcap_baselines.py`
- `dimension_stability/real_projected_bi_dimcap.py`
- `dimension_stability/ches_nonbi_metric_cell.py`
- `scripts/generate_paper_plots.py`

These scripts run the dimension-sweep and reduced-dimension non-BI diagnostics used for the CHES stability plot and appendix comparison plots.

## Local-family score-radius certificates

- `bi_scope_certificates/compute_bi_loc_margin_curves.py`
- `bi_scope_certificates/compute_full_suite_bi_loc.py`
- `bi_scope_certificates/build_bi_scope_tables.py`
- `scripts/plot_bi_loc_radius_curves.py`
- `scripts/build_full_suite_bi_loc_artifacts.py`
- `scripts/build_audited_score_radius_points.py`

These scripts compute and postprocess score-local `BI_loc` relaxed-margin certificates.

## ASCAD locality and trained-scope diagnostics

- `locality_diagnostics/ascad_temporal_window_sweep.py`
- `attacker_scope_diagnostics/attacker_scope_diagnostic.py`
- `attacker_scope_diagnostics/scope_cell_runner.py`
- `attacker_scope_diagnostics/aesrd_scope_seed_runner.py`
- `attacker_scope_diagnostics/evaluate_ascad_d0_challenger_fixed_split.py`

These scripts support the locality-radius, suite-size, and challenger/scope diagnostics.

## Known-posterior oracle diagnostics

- `bayes_oracle/bayes_oracle_synthetic.py`
- `bayes_oracle/gaussian_oracle_suite.py`
- `scripts/build_real_challenger_diagnostic.py`
- `scripts/merge_ascad_fixed_split_mlp_challengers.py`

The Bayes-oracle scripts cover synthetic known-posterior audits. The challenger scripts summarize real-data saturation diagnostics.
