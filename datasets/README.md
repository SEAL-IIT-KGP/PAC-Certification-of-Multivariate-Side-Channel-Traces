# datasets

Home of every downloaded trace set (`datasets/raw/`, ignored by git). `setup_datasets.py` fetches the two optional files that the other fetch scripts do not cover.

## The eight benchmarks

| Tag | Traces | Source |
|---|---|---|
| `ascad_desync_0`, `ascad_desync_50`, `ascad_desync_100` | ASCAD fixed key, desynchronization 0, 50, 100 | TCHES20 benchmark |
| `aes_hd`, `aes_rd`, `dpav4` | AES-HD, AES-RD, DPAv4 | TCHES20 benchmark |
| `ascad_random_key` | ASCAD random key, 1,400-sample window | ANSSI |
| `ches_ctf_2025` | CHES Challenge 2025 | challenge site |

## How to get each set

| Set | Command | Location |
|---|---|---|
| TCHES20 benchmark: six trace sets, loaders, 480 pretrained models (2.3 GB) | `paper_results/fetch_tches20.sh` | `tches20/TCHES20V3_CNN_SCA/` |
| ASCAD random key (420 MB) | `python -m paper_results.fetch_ascad_random_key` | `ascad_random_key/ASCAD.h5` |
| CHES-CTF-2025 (4 GB) | download from https://pace-tl.gitbook.io/ches-challenge-2025 | `ches_ctf_2025/CHES_Challenge.h5` |
| Raw ASCAD traces (6 GB) | `python -m multibyte_composition.fetch_raw_ascad` | `ascad_raw/ATMega8515_raw_traces.h5` |
| AES-RD input of the Transformer model | `python -m bi_scope_certificates.prepare_aesrd_estranet_h5` | `aes_rd/AES_RD_estranet_feature_standardized.h5` |
| Plain ASCAD and AES-HD files (optional, for `model_training/`) | `python -m datasets.setup_datasets --all --data-dir datasets/raw` | `ascad/ASCAD.h5`, `aes_hd/aes_hd_ext.npz` |

The EstraNet sources go to `external/EstraNet` (`paper_results/fetch_estranet.sh`). `python -m datasets.setup_datasets --info` lists what is present; `--verify --data-dir datasets/raw` checks the two optional files.

## TCHES20 paths

Scripts that use the TCHES20 loaders take the dataset root and the loader sources:

```bash
python -m dimension_stability.real_nonbi_dimcap_baselines \
  --tches20-dataset-dir datasets/raw/tches20/TCHES20V3_CNN_SCA/datasets \
  --tches20-src-dir datasets/raw/tches20/TCHES20V3_CNN_SCA/src
```
