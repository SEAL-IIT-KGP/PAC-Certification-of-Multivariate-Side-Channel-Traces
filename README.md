# Bounded Information Certification for Side-Channel Leakage

Code for **Bounded Information: PAC Certification of Multivariate Side-Channel Traces** (ASIACRYPT 2026), by Kuheli Pratihar, Nimish Mishra and Debdeep Mukhopadhyay (IIT Kharagpur).

Bounded Information (BI) is a guaranteed upper limit, in bits, on how often any attacker from a declared set of models guesses the key byte correctly. It is computed from a plain count of correct guesses on attack traces, with a KL-binomial interval and a union bound over the set. Unlike perceived information, hypothetical information and mutual-information estimators, it stays stable on long traces. The local extension `BI^loc(epsilon)` also covers every attacker whose scores stay within an audited distance of a set member.

This repository holds the certificate code, the evaluators for the paper's eight benchmarks, and a reproduction package that rebuilds every table and figure and checks each number against the printed one.

## Quick start

Everything runs in one Docker image. Build it and create the folders it shares with your machine:

```bash
git clone https://github.com/K1015/PAC-Certification-of-Multivariate-Side-Channel-Traces.git pac-artifact
cd pac-artifact
docker build -t bi-certification .
mkdir -p results datasets/raw checkpoints external
```

Level A rebuilds all tables and figures from the archived counts (no download, about a minute):

```bash
docker run --rm -v "$PWD/results:/artifact/results" bi-certification
```

For the other levels, define a shell function that shares all four folders:

```bash
artifact() {
  docker run --rm -v "$PWD/results:/artifact/results" -v "$PWD/datasets/raw:/artifact/datasets/raw" \
    -v "$PWD/checkpoints:/artifact/checkpoints" -v "$PWD/external:/artifact/external" \
    bi-certification "$@"
}
```

```bash
# Level B: TCHES20 traces and 480 pretrained models (0.5 GB)
artifact paper_results/fetch_tches20.sh
artifact python -m paper_results.reproduce tches20

# Level C: Transformer checkpoints (0.3 GB) and two trace files (0.4 GB and 4 GB)
tar -xzf bi_certification_checkpoints.tar.gz
artifact paper_results/fetch_estranet.sh
artifact python -m paper_results.fetch_ascad_random_key
# place CHES_Challenge.h5 (https://pace-tl.gitbook.io/ches-challenge-2025) in datasets/raw/ches_ctf_2025/
artifact python -m paper_results.reproduce estranet

# Level C, optional: the BI line of Figure 1 and the Transformer models of Figure 2
artifact python -m paper_results.reproduce estranet --suites ches_windows
artifact bash -c 'python -m bi_scope_certificates.prepare_aesrd_estranet_h5 &&
  for ds in ascad_random_key ascad_desync_50 ascad_desync_100 aes_rd; do
    python -m bi_scope_certificates.profile_epsilon_calibration --dataset $ds \
      --output-dir results/paper_results/recomputed/selected_centers/$ds; done &&
  python -m paper_results.reproduce report'

# Level D: the two full-version results (raw ASCAD traces, 4.4 GB)
artifact python -m paper_results.reproduce fullversion
```

Each run ends with one `[PASS]` line per table or figure and writes its outputs to `results/paper_results/`. Every run fits in 8 GB of memory, except the 7,000-sample window suite of Figure 1 (12 GB; raise the Docker Desktop limit first).

## Levels

| Level | Recomputes | Download | Time |
|---|---|---|---|
| A | nothing; checks the archived counts | none | 1 min |
| B | Tables 7–9 and Figure 2 from the TCHES20 traces and models | 0.5 GB | 8–15 min |
| C | Transformer rows of Tables 7–8 and the BI line of Figure 1 | 4.7 GB | 20–70 min |
| D | the two full-version tables | 4.4 GB | about 20 min |

All levels run on CPU. More detail:

- [`paper_results/README.md`](paper_results/README.md): each level, its output, and how to read the tables.
- [`docs/guides/`](docs/guides/README.md): one short PDF per directory with recorded terminal output; start with [`00_start_here.pdf`](docs/guides/pdf/00_start_here.pdf).
- [`docs/EXPERIMENT_MAP.md`](docs/EXPERIMENT_MAP.md): the script behind each table and figure.

## Hardware

- **Reproduction:** CPU only. The reference runs used a 24-core Linux workstation, a 16-core cluster node and a Mac (Apple M3 Pro).
- **Retraining (optional):** the Transformer checkpoints were trained on one CUDA GPU (5 to 8 hours per suite). [`bi_suite_8datasets/estranet_support/`](bi_suite_8datasets/estranet_support/README.md) has their settings and the retraining script; `bi_scope_certificates/train_keras_ascad_random_key.py` retrains the two Keras models in the same archive.
- The TCHES20 models are downloaded pretrained.

## Repository layout

Every directory has its own README.

| Directory | Contents |
|---|---|
| `core/` | certificate routines, PI/HI/MI estimators, dataset loading |
| `paper_results/` | reproduction driver, one builder per table and figure, archived counts |
| `bi_suite_8datasets/` | success counts of the model suites; `estranet_support/` for the Transformer checkpoints |
| `bi_scope_certificates/` | local certificates `BI^loc` and radius calibration |
| `dimension_stability/` | BI against PI, HI and MI as traces grow longer (Figure 1) |
| `epi_comparison/` | BI next to effective PI, with the temperature sweep (full version) |
| `multibyte_composition/` | four-byte key recovery on raw ASCAD traces (full version) |
| `bayes_oracle/`, `locality_diagnostics/`, `attacker_scope_diagnostics/` | appendix diagnostics |
| `model_training/` | small MLP, CNN and Transformer training examples |
| `datasets/`, `scripts/`, `docs/` | trace layout, full-version plots, experiment map and guides |

## Setup without Docker

Python 3.11, from the repository root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                                                          # Level A
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-ml.txt  # TensorFlow, PyTorch
```

`requirements-lock.txt` is the full environment of the Docker image. Both setups run on x86-64 Linux and on Apple Silicon. Run every script as `python -m package.script` from the repository root.

## External code and data

- TCHES20 loaders and pretrained models: https://github.com/KULeuven-COSIC/TCHES20V3_CNN_SCA
- EstraNet: https://github.com/suvadeep-iitb/EstraNet
- Trace sets: [`datasets/README.md`](datasets/README.md)

The fetch scripts in `paper_results/` download code and data at fixed versions.

## License

MIT (see `LICENSE`). The TCHES20 benchmark, the EstraNet sources and the datasets keep their authors' terms.

## Citation

```bibtex
@inproceedings{pratihar2026bounded,
  title     = {Bounded Information: {PAC} Certification of Multivariate Side-Channel Traces},
  author    = {Pratihar, Kuheli and Mishra, Nimish and Mukhopadhyay, Debdeep},
  booktitle = {Advances in Cryptology -- {ASIACRYPT} 2026},
  year      = {2026}
}
```
