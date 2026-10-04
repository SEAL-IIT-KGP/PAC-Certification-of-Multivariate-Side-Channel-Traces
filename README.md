# Bounded Information Certification for Side-Channel Leakage

Code for **Bounded Information: PAC Certification of Multivariate Side-Channel Traces** (ASIACRYPT 2026), by Kuheli Pratihar, Nimish Mishra and Debdeep Mukhopadhyay (Indian Institute of Technology Kharagpur).

**What it is.** A side-channel attacker trains a model on traces with known keys and uses it to guess the key byte behind fresh traces. Bounded Information (BI) is a guaranteed upper limit on how often any attacker from a declared set can guess right. It is computed from a plain count (how many attack traces a model classifies correctly), turned into a limit that holds with high confidence (a KL-binomial interval with a union bound over the set), and reported in bits. It stays stable where perceived information, hypothetical information and mutual-information estimators break down on long traces. The local extension `BI^loc(epsilon)` also covers every attacker whose scores stay within an audited distance of a set member.

**What is here.** The certificate code, the evaluators for the eight benchmarks of the paper, and a reproduction package that rebuilds every table and figure and checks each number against the printed one.

## Quick start

Build the Docker image and run the default check. It rebuilds all tables and figures from the archived counts, needs no download, and takes about a minute:

```bash
docker build -t bi-certification .
docker run --rm -v "$PWD/results:/artifact/results" bi-certification
```

Without Docker: `pip install -r requirements.txt && python -m paper_results.reproduce archived`.

## Reproducing the paper

`paper_results/reproduce.py` has four levels. Each level recomputes more of the inputs from public data and reports which printed values match.

| Level | Command | Recomputes | Download | Hardware | Time |
|---|---|---|---|---|---|
| A | `python -m paper_results.reproduce archived` | nothing; checks the archived counts | none | any laptop | 1 min |
| B | `python -m paper_results.reproduce tches20` | the inputs of Tables 7, 8, 9 and Figure 2 from the public TCHES20 traces and its 480 pretrained models | 0.5 GB | 8 GB RAM, CPU | 8 to 15 min on 16 to 24 cores |
| C | `python -m paper_results.reproduce estranet` | the Transformer rows of Tables 7 and 8 and the BI line of Figure 1 from the released checkpoints | 0.3 GB checkpoints + 4.4 GB traces | 16 GB RAM, CPU | 20 to 70 min |
| D | `python -m paper_results.reproduce fullversion` | the full-version tables from the raw ASCAD traces and the TCHES20 models | 4.4 GB | 8 GB RAM, CPU, PyTorch | about 20 min on 24 cores |

Where to read next:

- [`paper_results/README.md`](paper_results/README.md): the four levels in detail, the expected output, and how to read each table.
- [`docs/guides/`](docs/guides/README.md): one short PDF per directory for a first-time reader, with the terminal output of every command recorded inside the Docker image on a Linux workstation (x86-64, CPU only); guide 00 also lists the times on a Mac (Apple M3 Pro). Start with [`00_start_here.pdf`](docs/guides/pdf/00_start_here.pdf).
- [`docs/EXPERIMENT_MAP.md`](docs/EXPERIMENT_MAP.md): which script produces each table and figure.

## Hardware: CPU to reproduce, GPU only to train

All four levels run on CPU: they load pretrained models and count successes. The reference runs used a 24-core workstation, a 16-core cluster node, and a laptop. A GPU is needed only to **train** models, which no level does:

- The EstraNet Transformer checkpoints (Transformer rows of Tables 7 and 8, the reference models of Figure 2, the window suites of Figure 1) were trained on CUDA GPUs with the upstream `train_trans.py` for up to 4,000,000 steps at batch size 16. They are distributed, so no retraining is needed; their settings are in [`bi_suite_8datasets/estranet_support/README.md`](bi_suite_8datasets/estranet_support/README.md).
- The TCHES20 models are downloaded pretrained.
- The training examples in `model_training/` run on CPU for the MLP and the small CNN; the Transformer example benefits from a GPU.

One number depends on the platform: the calibrated radius of Figure 2 covers the score variation of the Transformers under repeated GPU inference (up to 1.9e-4). On CPU that variation is about zero, and the builders keep the archived GPU profile so that the printed radius is reproduced either way (see the notes in `paper_results/README.md`).

## Repository layout

Every directory has its own README: what it does, its files, what to have in place, the commands, the outputs.

| Directory | Contents |
|---|---|
| `core/` | certificate routines, baseline estimators (PI, HI, MI), dataset loading |
| `paper_results/` | reproduction driver, one builder per table and figure, archived counts, printed values, fetch scripts |
| `bi_suite_8datasets/` | success counts of the model suites on the eight benchmarks; `estranet_support/` adds the CHES-CTF-2025 loader and the training settings of the Transformer checkpoints |
| `bi_scope_certificates/` | local score-family certificates `BI^loc`, radius calibration, score-radius curves |
| `multibyte_composition/` | four-byte key recovery on the raw ASCAD traces (full version) |
| `epi_comparison/` | BI versus effective PI on the same model, with the temperature sweep (full version) |
| `dimension_stability/` | BI against PI, HI and MI as the trace dimension grows (the estimator cells of Figure 1) |
| `locality_diagnostics/`, `attacker_scope_diagnostics/`, `bayes_oracle/` | the temporal-window, attacker-scope and synthetic-oracle diagnostics of the appendix |
| `model_training/` | small MLP, CNN and Transformer training examples (not the paper's models) |
| `datasets/` | layout and sources of the trace sets |
| `scripts/` | postprocessing and the plots of the full version |
| `docs/` | the experiment map and the guides |

## Setup

Python 3.11 and the versions pinned in the requirement files:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-ml.txt
```

- `requirements.txt`: the certificate code and the table and figure builders (enough for Level A).
- `requirements-ml.txt`: TensorFlow 2.15 (pretrained Keras models, EstraNet checkpoints) and PyTorch 2.2 (diagnostics, training examples).
- `requirements-lock.txt`: the complete environment of the reference runs, which the `Dockerfile` reproduces.

The image and the virtual environment have been built and run on x86-64 Linux and on Apple Silicon (arm64 Linux inside Docker, native macOS Python); the requirement files select the matching wheels. Run every entry point from the repository root as `python -m package.script`, because the scripts import shared code as `core.*`. The code is formatted and linted with `ruff` (`pyproject.toml`).

## External code and data

- TCHES20 loaders and pretrained models: https://github.com/KULeuven-COSIC/TCHES20V3_CNN_SCA
- EstraNet: https://github.com/suvadeep-iitb/EstraNet (used unchanged, plus the CHES-CTF-2025 loader in `estranet_support/`)
- Trace sets: [`datasets/README.md`](datasets/README.md) lists each benchmark's source and expected file.

`paper_results/fetch_tches20.sh`, `paper_results/fetch_estranet.sh` and `paper_results/fetch_ascad_random_key.py` fetch the code and data at fixed versions.

## License

MIT (see `LICENSE`). The TCHES20 benchmark, the EstraNet sources and the datasets are distributed by their authors under their own terms.

## Citation

```bibtex
@inproceedings{pratihar2026bounded,
  title     = {Bounded Information: {PAC} Certification of Multivariate Side-Channel Traces},
  author    = {Pratihar, Kuheli and Mishra, Nimish and Mukhopadhyay, Debdeep},
  booktitle = {Advances in Cryptology -- {ASIACRYPT} 2026},
  year      = {2026}
}
```
