# Dataset Setup

Place raw or externally prepared datasets under `datasets/raw/`. Keep derived arrays, caches, and downloaded archives out of version control.

The paper uses eight dataset rows:

- `ascad_desync_0`: ASCAD fixed-key traces with no desynchronization.
- `ascad_desync_50`: ASCAD fixed-key traces with desynchronization 50.
- `ascad_desync_100`: ASCAD fixed-key traces with desynchronization 100.
- `ascad_random_key`: ASCAD random-key traces.
- `aes_hd`: AES-HD traces.
- `aes_rd`: AES-RD traces.
- `dpav4`: DPAv4 traces.
- `ches_ctf_2025`: CHES CTF 2025 traces.

Suggested layout:

```text
datasets/raw/
  ascad/
  ascad_random_key/
  aes_hd/
  tches20/
    TCHES20V3_CNN_SCA/
      datasets/
      src/
  ches_ctf_2025/
```

Use `setup_datasets.py` for datasets that can be fetched or verified directly:

```bash
python -m datasets.setup_datasets --info
python -m datasets.setup_datasets --verify --data-dir datasets/raw
python -m datasets.setup_datasets --ascad --data-dir datasets/raw
python -m datasets.setup_datasets --aes-hd --data-dir datasets/raw
```

For the TCHES20-style loaders used by `ascad_desync_0`, `ascad_desync_50`, `ascad_desync_100`, `aes_rd`, and `dpav4`, pass both the dataset root and the external loader source directory to scripts:

```bash
python -m dimension_stability.real_nonbi_dimcap_baselines \
  --data-dir datasets/raw/tches20/TCHES20V3_CNN_SCA/datasets \
  --tches20-src-dir datasets/raw/tches20/TCHES20V3_CNN_SCA/src
```

`ascad_random_key` and `ches_ctf_2025` are evaluated through checkpoint-suite scripts. Store the dataset and checkpoint roots locally and pass them with the corresponding script arguments.
