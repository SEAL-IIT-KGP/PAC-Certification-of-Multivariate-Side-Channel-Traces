# Bounded Information Certification for Side-Channel Leakage

Implementation code for **Bounded Information: PAC Certification of Multivariate Side-Channel Traces** (ASIACRYPT 2026), by Kuheli Pratihar, Nimish Mishra, and Debdeep Mukhopadhyay (Indian Institute of Technology Kharagpur).

Side-channel leakage certification asks how much an implementation leaks and how powerful an attacker must be to exploit that leakage. Existing information-theoretic estimators, such as perceived information (PI), hypothetical information (HI), and nonparametric mutual information (MI), aim to quantify distributional leakage, but they become unstable in high-dimensional traces and do not provide a finite-sample certificate of the best attacker. We introduce *Bounded Information* (BI), a Probably Approximately Correct (PAC) leakage-certification methodology that upper-bounds the population exact-recovery success of a declared attacker scope on fresh traces. The attacker suite certificate, $\mathrm{BI}^{\mathrm{suite}}$, applies a KL-binomial confidence interval with a union bound over a fixed suite that contains every trained attacker, preprocessing choice, hyperparameter, and seed to be reported. BI therefore turns standard profiled-attack evaluation into an auditable certificate with an explicit attacker scope and confidence level. We further define $\mathrm{BI}^{\mathrm{loc}}(\varepsilon)$, a local score-family extension that certifies every attacker whose normalized score function stays within an audited radius of a suite member, and this extension stays a binomial quantity because the radius enters only through a relaxed-margin count. Across eight side-channel benchmarks with trace dimensions up to $7{,}000$, BI provides stable certificates without density estimation, and its tightness diagnostics tell an evaluator whether a certified value is a genuine measurement of leakage or a conservative bound that more attack traces would tighten.

This repository provides tools for computing certifiable leakage bounds from trained models and datasets in side-channel analysis, addressing the instability of traditional metrics (PI/HI/MI) in high-dimensional settings.

## Reproducing the Paper's Results

`paper_results/` rebuilds Tables 7-9 and Figures 1-2 of the paper and compares every regenerated value with the printed one. The quickest check needs no data and takes about a minute:

```bash
docker build -t bi-certification .
docker run --rm -v "$PWD/results:/artifact/results" bi-certification
```

or, without Docker, `pip install -r requirements.txt && python -m paper_results.reproduce archived`. The same driver recomputes the results from the public TCHES20 traces and pretrained models (`python -m paper_results.reproduce tches20`) and from the released EstraNet checkpoints. `paper_results/README.md` is the step-by-step guide, with the expected output of each level, and `docs/EXPERIMENT_MAP.md` maps every table and figure to its code.

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

### `paper_results/`

Table and figure builders for the paper, the archived per-model counts they start from, the values printed in the paper, and the reproduction driver:

- `reproduce.py`: runs a reproduction level (`archived`, `tches20`, `estranet`) and reports which results match the paper.
- `table7_bi_suite.py`, `table8_key_recovery.py`, `table9_biloc_full_suite.py`, `figure1_dimension_sweep.py`, `figure2_biloc_selected_centers.py`: one builder per result.
- `evaluate_selected_models.py`, `evaluate_estranet_suites.py`: EPI and key recovery for the selected suite members, and the EstraNet suite counts.
- `fetch_tches20.sh`, `fetch_estranet.sh`: fetch the public benchmark and the EstraNet sources.

### `bi_suite_8datasets/`

Finite-suite certificate evaluation on the eight paper datasets:

- `evaluate_tches20_pretrained_fixed_split.py`: TCHES20-style pretrained fixed-split suites (use `--split attack` for certificates).
- `evaluate_estranet_bi_suite.py`: EstraNet checkpoint-suite evaluation.
- `summarize_ascad_estranet_bi_baselines.py` and `summarize_ches_transformer_bi_sweep.py`: summary builders for checkpoint-suite rows.

### `bi_scope_certificates/`

Score-local certificate code for `BI_loc`:

- `compute_bi_loc_margin_curves.py`: relaxed-margin score-radius curves.
- `compute_full_suite_bi_loc.py`: full-suite local-family computation.
- `profile_epsilon_calibration.py`: calibration/certification split and score-variation profile behind the calibrated radius.
- `prepare_aesrd_estranet_h5.py`: AES-RD input file for the EstraNet reference model.
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

### `multibyte_composition/`

Four-byte key recovery on the raw ASCAD traces (full version): point-of-interest scan, per-byte attackers, and exact four-byte composition.

### `epi_comparison/`

Same-model comparison of BI with effective PI on the public TCHES20 models, including the temperature-rescaling sweep (full version).

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

The repository was tested with Python 3.11 and the versions pinned in the requirement files:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-ml.txt
```

`requirements.txt` (NumPy 1.26.4, SciPy 1.13.1, pandas 2.2.3, scikit-learn 1.5.2, Matplotlib 3.9.4, h5py 3.11.0, tqdm 4.66.5) covers the certificate code and the table and figure builders. `requirements-ml.txt` adds TensorFlow 2.15.0, which evaluates the pretrained Keras models and the EstraNet checkpoints, and PyTorch 2.2.2, which is used by the diagnostics and the training examples. CPU inference is sufficient for the evaluations behind the paper's tables. `requirements-lock.txt` lists the complete resolved environment of the reference runs (Linux x86-64), and the `Dockerfile` builds an image with both requirement files installed.

Run all entry points from the repository root (for example `python -m bi_suite_8datasets.evaluate_estranet_bi_suite --help`), because the scripts import shared code as `core.*`.

The code is formatted and linted with `ruff` using the settings in `pyproject.toml` (120 columns); `ruff format . && ruff check .` keeps a contribution consistent with the rest.

## External dependencies

Some evaluators import code and load models that are not part of this repository:

- TCHES20 dataset loaders (`dataLoaders`) and pretrained models: https://github.com/KULeuven-COSIC/TCHES20V3_CNN_SCA. Pass the loader source directory with `--tches20-src-dir` where a script asks for it.
- EstraNet (`data_utils`, `transformer`): https://github.com/suvadeep-iitb/EstraNet. The default location is `external/EstraNet`.

`paper_results/fetch_tches20.sh` and `paper_results/fetch_estranet.sh` fetch both at fixed commits. This repository uses the two code bases unchanged and adds a CHES-CTF-2025 loader (`data_utils_ches25`) next to the EstraNet sources; `bi_suite_8datasets/estranet_support/README.md` describes the added files and the training settings of the EstraNet checkpoints.

See `datasets/README.md` for the dataset files each benchmark expects.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The TCHES20 benchmark, the EstraNet sources, and the datasets are distributed by their authors under their own terms and are fetched from their original locations.

## Citation

```bibtex
@inproceedings{pratihar2026bounded,
  title     = {Bounded Information: {PAC} Certification of Multivariate Side-Channel Traces},
  author    = {Pratihar, Kuheli and Mishra, Nimish and Mukhopadhyay, Debdeep},
  booktitle = {Advances in Cryptology -- {ASIACRYPT} 2026},
  year      = {2026}
}
```
