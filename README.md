# Bounded Information Certification for Side-Channel Leakage

Implementation code for **Bounded Information: PAC Certification of Multivariate Side-Channel Traces** (ASIACRYPT 2026), by Kuheli Pratihar, Nimish Mishra, and Debdeep Mukhopadhyay (Indian Institute of Technology Kharagpur).

Side-channel leakage certification asks how much an implementation leaks and how powerful an attacker must be to exploit that leakage. Existing information-theoretic estimators, such as perceived information (PI), hypothetical information (HI), and nonparametric mutual information (MI), aim to quantify distributional leakage, but they become unstable in high-dimensional traces and do not provide a finite-sample certificate of the best attacker. We introduce *Bounded Information* (BI), a Probably Approximately Correct (PAC) leakage-certification methodology that upper-bounds the population exact-recovery success of a declared attacker scope on fresh traces. The attacker suite certificate, $\mathrm{BI}^{\mathrm{suite}}$, applies a KL-binomial confidence interval with a union bound over a fixed suite that contains every trained attacker, preprocessing choice, hyperparameter, and seed to be reported. BI therefore turns standard profiled-attack evaluation into an auditable certificate with an explicit attacker scope and confidence level. We further define $\mathrm{BI}^{\mathrm{loc}}(\varepsilon)$, a local score-family extension that certifies every attacker whose normalized score function stays within an audited radius of a suite member, and this extension stays a binomial quantity because the radius enters only through a relaxed-margin count. Across eight side-channel benchmarks with trace dimensions up to $7{,}000$, BI provides stable certificates without density estimation, and its tightness diagnostics tell an evaluator whether a certified value is a genuine measurement of leakage or a conservative bound that more attack traces would tighten.

This repository provides tools for computing certifiable leakage bounds from trained models and datasets in side-channel analysis, addressing the instability of traditional metrics (PI/HI/MI) in high-dimensional settings.

## Repository Layout

### `core/`

Shared implementation utilities:

- `bi_certificate.py`: KL-binomial BI certificate routines.
- `baseline_metrics.py`: PI, HI, MI, and related baseline estimators.
- `data_loader.py`: dataset loading, synthetic data generation, and split helpers.

### `datasets/`

Dataset setup documentation and lightweight setup helpers. See `datasets/README.md` for the expected layout of the eight benchmark datasets.

### `model_training/`

Illustrative training examples for the attacker families. They are not the pretrained TCHES20 or EstraNet reference models evaluated in the paper, and their checkpoints are not directly loadable by the BI evaluators:

- `common.py`: shared data loading, windowing, and split helpers for the examples.
- `train_mlp_example.py`: MLP profiled attacker.
- `train_cnn_example.py`: 1D CNN trace classifier.
- `train_transformer_example.py`: Transformer-style attacker for longer traces.

### `bi_suite_8datasets/`

Finite-suite certificate evaluation on the eight paper datasets:

- `evaluate_tches20_pretrained_fixed_split.py`: TCHES20-style pretrained fixed-split suites (use `--split attack` for certificates).
- `evaluate_estranet_bi_suite.py`: EstraNet checkpoint-suite evaluation.
- `summarize_ascad_estranet_bi_baselines.py` and `summarize_ches_transformer_bi_sweep.py`: summary builders for checkpoint-suite rows.

### `bi_scope_certificates/`

Score-local certificate code for `BI_loc`:

- `compute_bi_loc_margin_curves.py`: relaxed-margin score-radius curves.
- `compute_full_suite_bi_loc.py`: full-suite local-family computation.
- `build_bi_scope_tables.py`: table assembly for local and suite certificates.

### `dimension_stability/`

Diagnostics comparing BI against PI/HI/MI-style estimators under growing trace dimension:

- `multivariate_stability_driver.py`: synthetic multivariate stability driver.
- `real_nonbi_dimcap_baselines.py`: reduced-dimension non-BI baselines.
- `real_projected_bi_dimcap.py`: projected-dimension BI stability diagnostics.
- `ches_nonbi_metric_cell.py`: CHES-CTF high-dimensional non-BI metric cells.

### `locality_diagnostics/`

Temporal-window locality diagnostics:

- `ascad_temporal_window_sweep.py`: ASCAD temporal-window BI sweep.

### `attacker_scope_diagnostics/`

Trained attacker-scope and challenger diagnostics:

- `attacker_scope_diagnostic.py`: main scope comparison driver.
- `scope_cell_runner.py`, `aesrd_scope_seed_runner.py`, and `evaluate_ascad_d0_challenger_fixed_split.py`: focused rerun and challenger wrappers.

### `bayes_oracle/`

Synthetic known-posterior oracle diagnostics:

- `gaussian_oracle_suite.py`: byte-scale Gaussian oracle suite.
- `bayes_oracle_synthetic.py`: masked/locality synthetic oracle diagnostic.

### `scripts/`

Postprocessing and plot generation:

- `build_bi_suite_paper_artifacts.py`: finite-suite BI table assembly.
- `audit_bi_suite_margin_reduction.py`: margin-reduction audit of the finite-suite rows.
- `build_full_suite_bi_loc_artifacts.py` and `build_audited_score_radius_points.py`: local-family postprocessing.
- `plot_bi_loc_radius_curves.py`: score-radius plot generation.
- `generate_paper_plots.py`: plot regeneration from result CSVs (see `scripts/README.md` for required inputs).
- `build_real_challenger_diagnostic.py` and `merge_ascad_fixed_split_mlp_challengers.py`: real-data challenger summaries.

### `docs/`

Manuscript-to-code mapping. Start with `docs/EXPERIMENT_MAP.md` when locating the code behind a paper result.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

TensorFlow and PyTorch are optional at install time, but scripts that evaluate or train neural checkpoints need the matching framework installed.

Run all entry points from the repository root (for example `python -m bi_suite_8datasets.evaluate_estranet_bi_suite --help`), because the scripts import shared code as `core.*`.

## External dependencies

Some evaluators import code and load models that are not part of this repository:

- TCHES20 dataset loaders (`dataLoaders`) and pretrained models: https://github.com/KULeuven-COSIC/TCHES20V3_CNN_SCA. Pass the loader source directory with `--tches20-src-dir` where a script asks for it.
- EstraNet (`data_utils`, `data_utils_ches25`, `transformer`): https://github.com/suvadeep-iitb/EstraNet. The default location is `external/EstraNet`.

See `datasets/README.md` for the dataset files each benchmark expects.

## Citation

```bibtex
@inproceedings{pratihar2026bounded,
  title     = {Bounded Information: {PAC} Certification of Multivariate Side-Channel Traces},
  author    = {Pratihar, Kuheli and Mishra, Nimish and Mukhopadhyay, Debdeep},
  booktitle = {Advances in Cryptology -- {ASIACRYPT} 2026},
  year      = {2026}
}
```
