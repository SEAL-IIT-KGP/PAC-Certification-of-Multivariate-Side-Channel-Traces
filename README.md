# Bounded Information Certification for Side-Channel Leakage

Code for **Bounded Information: PAC Certification of Multivariate Side-Channel Traces** (ASIACRYPT 2026), by Kuheli Pratihar, Nimish Mishra, and Debdeep Mukhopadhyay (Indian Institute of Technology Kharagpur).

Bounded Information (BI) is a PAC leakage certificate: it upper-bounds the exact-recovery success of a declared attacker scope on fresh traces with a KL-binomial confidence interval and a union bound over the attacker suite, and it stays stable where perceived information, hypothetical information, and mutual-information estimators break down in high-dimensional traces. The local extension $\mathrm{BI}^{\mathrm{loc}}(\varepsilon)$ certifies every attacker whose score function lies within an audited radius of a suite member. This repository holds the certificate code, the evaluators for the eight benchmarks of the paper, and a reproduction package that rebuilds every table and figure and checks it against the printed values.

## Quick start

The fastest check rebuilds all tables and figures from the archived per-model counts, needs no data download, and takes about a minute:

```bash
docker build -t bi-certification .
docker run --rm -v "$PWD/results:/artifact/results" bi-certification
```

Without Docker: `pip install -r requirements.txt && python -m paper_results.reproduce archived`.

## Reproducing the paper

`paper_results/reproduce.py` has four levels. Each one recomputes more of the inputs from public data and reports which printed values match.

| Level | Command | Recomputes | Download | Hardware | Time |
|---|---|---|---|---|---|
| A | `python -m paper_results.reproduce archived` | nothing; checks the archived counts | none | any laptop | 1 min |
| B | `python -m paper_results.reproduce tches20` | Table 7, 8, 9 and Figure 2 inputs from the public TCHES20 traces and its 480 pretrained models | 0.5 GB | 8 GB RAM, CPU | 8–15 min on 16–24 cores |
| C | `python -m paper_results.reproduce estranet` | the EstraNet Transformer rows of Tables 7–8 and the BI line of Figure 1 from the released checkpoints | 0.3 GB checkpoints + 4.4 GB traces | 16 GB RAM, CPU | 20–70 min |
| D | `python -m paper_results.reproduce fullversion` | the full-version tables from the raw ASCAD traces and TCHES20 models | 4.4 GB | 8 GB RAM, CPU, PyTorch | about 20 min on 24 cores |

[`paper_results/README.md`](paper_results/README.md) is the step-by-step guide with the expected output of every level, and [`docs/EXPERIMENT_MAP.md`](docs/EXPERIMENT_MAP.md) maps each table and figure to its script.

## Hardware: CPU for reproduction, GPU for training

Everything behind the paper's tables and figures is inference and counting, and all four levels above run on CPU only; the reference runs used a 24-core workstation and a 16-core cluster node. A GPU is needed only to **train** models:

- **EstraNet Transformer checkpoints** (the Transformer rows of Tables 7–8, the reference models of Figure 2, and the window suites of Figure 1) were trained on CUDA GPUs with the upstream `train_trans.py` for up to 4,000,000 steps at batch size 16; retraining them on CPU is impractical. The settings of every released checkpoint are listed in [`bi_suite_8datasets/estranet_support/README.md`](bi_suite_8datasets/estranet_support/README.md), and the checkpoints themselves are distributed so that no retraining is needed.
- **The TCHES20 models** are downloaded pretrained; nothing is trained at Levels A–D.
- **The training examples** in `model_training/` run on CPU for the MLP and small CNN settings; the Transformer example benefits from a GPU.

One number depends on the platform: the calibrated radius $\varepsilon$ covers the score variation of the Transformers under repeated GPU inference (up to 1.9e-4). On CPU that variation is about zero, and the builders keep the archived GPU profile so that the printed radius is reproduced either way (see the scope notes in `paper_results/README.md`).

## Repository layout

Every directory has its own README with the commands it supports.

| Directory | Contents |
|---|---|
| `core/` | BI certificate routines, baseline estimators (PI, HI, MI), dataset loading |
| `paper_results/` | reproduction driver, one builder per table and figure, archived counts, printed values, fetch scripts |
| `bi_suite_8datasets/` | finite-suite certificates on the eight benchmarks; `estranet_support/` adds the CHES-CTF-2025 loader and training notes for EstraNet |
| `bi_scope_certificates/` | local score-family certificates $\mathrm{BI}^{\mathrm{loc}}$, radius calibration, score-radius curves |
| `multibyte_composition/` | four-byte key recovery on the raw ASCAD traces (full version) |
| `epi_comparison/` | same-model BI versus effective PI with the temperature sweep (full version) |
| `dimension_stability/` | BI against PI, HI and MI under growing trace dimension (Figure 1 estimators) |
| `locality_diagnostics/`, `attacker_scope_diagnostics/`, `bayes_oracle/` | temporal-window, attacker-scope and synthetic-oracle diagnostics of the appendix |
| `model_training/` | illustrative MLP, CNN and Transformer training examples (not the paper's reference models) |
| `datasets/` | expected layout and sources of the benchmark datasets |
| `scripts/` | postprocessing and plot generation from result CSVs |
| `docs/` | manuscript-to-code map |

## Setup

Tested with Python 3.11 and the versions pinned in the requirement files:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-ml.txt
```

`requirements.txt` covers the certificate code and the table and figure builders; `requirements-ml.txt` adds TensorFlow 2.15 (pretrained Keras models, EstraNet checkpoints) and PyTorch 2.2 (diagnostics, training examples); `requirements-lock.txt` is the complete environment of the reference runs, which the `Dockerfile` reproduces. Run every entry point from the repository root (`python -m package.script`), because the scripts import shared code as `core.*`. The code is formatted and linted with `ruff` (`pyproject.toml`).

## External code and data

- TCHES20 loaders and pretrained models: https://github.com/KULeuven-COSIC/TCHES20V3_CNN_SCA
- EstraNet: https://github.com/suvadeep-iitb/EstraNet (used unchanged, plus the CHES-CTF-2025 loader in `estranet_support/`)
- Datasets: see [`datasets/README.md`](datasets/README.md) for each benchmark's source and expected file

`paper_results/fetch_tches20.sh`, `paper_results/fetch_estranet.sh` and `paper_results/fetch_ascad_random_key.py` fetch the code and data at fixed versions.

## License

MIT (see `LICENSE`). The TCHES20 benchmark, the EstraNet sources, and the datasets are distributed by their authors under their own terms.

## Citation

```bibtex
@inproceedings{pratihar2026bounded,
  title     = {Bounded Information: {PAC} Certification of Multivariate Side-Channel Traces},
  author    = {Pratihar, Kuheli and Mishra, Nimish and Mukhopadhyay, Debdeep},
  booktitle = {Advances in Cryptology -- {ASIACRYPT} 2026},
  year      = {2026}
}
```
