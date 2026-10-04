# datasets

**What this folder does.** It is the home of every downloaded trace set (`datasets/raw/`, ignored by git) and holds `setup_datasets.py`, which fetches or verifies the two sets that the other fetch scripts do not cover.

## The eight benchmarks of the paper

| Tag | Traces | Source |
|---|---|---|
| `ascad_desync_0`, `ascad_desync_50`, `ascad_desync_100` | ASCAD fixed key, desynchronization 0, 50, 100 | TCHES20 checkout |
| `aes_hd` | AES-HD | TCHES20 checkout |
| `aes_rd` | AES-RD | TCHES20 checkout |
| `dpav4` | DPAv4 | TCHES20 checkout |
| `ascad_random_key` | ASCAD random key, 1,400-sample window | ANSSI release |
| `ches_ctf_2025` | CHES Challenge 2025 | the challenge site |

## Layout

```text
datasets/raw/
  ascad/                 ASCAD.h5 (plain ASCAD file, optional)
  ascad_random_key/      ASCAD.h5
  ascad_raw/             ATMega8515_raw_traces.h5 (Level D)
  aes_hd/                aes_hd_ext.npz (optional)
  tches20/
    TCHES20V3_CNN_SCA/   datasets/, src/, models/
  ches_ctf_2025/         CHES_Challenge.h5
```

Other default paths read by the scripts: `datasets/raw/aes_rd/AES_RD_estranet.h5` and `datasets/raw/aes_rd/AES_RD_estranet_feature_standardized.h5` (written by `bi_scope_certificates/prepare_aesrd_estranet_h5.py`).

## How to get each set

| Set | Command or source |
|---|---|
| TCHES20 benchmark: six trace sets, loaders and 480 pretrained models (about 2.3 GB unpacked) | `paper_results/fetch_tches20.sh` clones https://github.com/KULeuven-COSIC/TCHES20V3_CNN_SCA at a fixed commit into `datasets/raw/tches20/TCHES20V3_CNN_SCA` and unpacks it |
| EstraNet sources | `paper_results/fetch_estranet.sh` clones https://github.com/suvadeep-iitb/EstraNet into `external/EstraNet` |
| ASCAD random key | `python -m paper_results.fetch_ascad_random_key` (the `ASCAD.h5` of https://github.com/ANSSI-FR/ASCAD, `ATMEGA_AES_v1/ATM_AES_v1_variable_key`) |
| CHES-CTF-2025 | manual download of `CHES_Challenge.h5` from https://pace-tl.gitbook.io/ches-challenge-2025 into `datasets/raw/ches_ctf_2025/` |
| Raw ASCAD traces | `python -m multibyte_composition.fetch_raw_ascad` |
| Plain ASCAD and AES-HD files (optional, for `model_training/`) | `setup_datasets.py`, below |

```bash
python -m datasets.setup_datasets --info
python -m datasets.setup_datasets --verify --data-dir datasets/raw
python -m datasets.setup_datasets --ascad --data-dir datasets/raw
python -m datasets.setup_datasets --aes-hd --data-dir datasets/raw
```

`--verify` checks only the two optional files and exits non-zero when either is missing; that is fine for all four levels of `paper_results/`.

## Passing the TCHES20 paths to scripts

For the TCHES20-style loaders used by `ascad_desync_0`, `ascad_desync_50`, `ascad_desync_100`, `aes_rd` and `dpav4`, scripts take both the dataset root and the loader source directory:

```bash
python -m dimension_stability.real_nonbi_dimcap_baselines \
  --tches20-dataset-dir datasets/raw/tches20/TCHES20V3_CNN_SCA/datasets \
  --tches20-src-dir datasets/raw/tches20/TCHES20V3_CNN_SCA/src
```

`ascad_random_key` and `ches_ctf_2025` are evaluated through the checkpoint-suite scripts, which take the file and checkpoint paths as arguments. Keep derived arrays, caches and archives out of version control.
