# estranet_support

The Transformer attackers in the paper are EstraNet models (https://github.com/suvadeep-iitb/EstraNet, commit `7f969ce`). The evaluators use the upstream sources unchanged; this folder adds the CHES-CTF-2025 loader, the training patch and the retraining script.

## Files

| File | Purpose |
|---|---|
| `data_utils_ches25.py` | loader for `CHES_Challenge.h5`: byte 0 of the AES S-box output, with notch denoising, mean alignment and per-sample standardization (the scheme of https://github.com/LucianoXu/CHES2025-Challenge) |
| `evaluation_utils_ches25.py` | key-rank evaluation for the same dataset, used by the training script |
| `train_trans.patch` | adds the `CHES25` dataset and a `rank_repeats` flag to the upstream `train_trans.py` (training only) |
| `train_checkpoints.sh` | retrains one checkpoint directory of the release with the settings below |

`paper_results/fetch_estranet.sh` clones EstraNet into `external/EstraNet` and adds the two loader files; `train_checkpoints.sh` also applies the patch.

## Retraining (optional, GPU)

The released checkpoints make retraining unnecessary. A full run is 4,000,000 steps at batch size 16, which takes 5 to 8 hours per suite on one CUDA GPU.

Place the suite's trace file as for Level C (`paper_results/README.md`), install `requirements-ml-gpu.txt` instead of `requirements-ml.txt` (TensorFlow 2.15 with CUDA, Linux x86-64), and run from the repository root:

```bash
bi_suite_8datasets/estranet_support/train_checkpoints.sh ascadr
```

With Docker (Linux x86-64 with the NVIDIA driver and the NVIDIA Container Toolkit), build the GPU image and run it with `--gpus all`:

```bash
docker build --build-arg TF=gpu -t bi-certification-gpu .
docker run --rm --gpus all -v "$PWD/results:/artifact/results" -v "$PWD/datasets/raw:/artifact/datasets/raw" \
  -v "$PWD/checkpoints:/artifact/checkpoints" -v "$PWD/external:/artifact/external" \
  bi-certification-gpu bi_suite_8datasets/estranet_support/train_checkpoints.sh ascadr
```

- The script prints the number of GPUs TensorFlow sees (`gpus=1`) before training starts.
- Checkpoints go to `checkpoints/estranet_retrained/<suite>/trans_long-<index>`: index 1 is the model at step 0, and each later index adds 20,000 steps (index 201 ends a 4,000,000-step run).
- Running the same command again resumes from the latest checkpoint.
- To evaluate the retrained checkpoints, pass `--checkpoint-root checkpoints/estranet_retrained` to `paper_results/evaluate_estranet_suites.py` or `bi_scope_certificates/profile_epsilon_calibration.py`.

## Training settings of the released checkpoints

All runs use `train_trans.py` with `n_layer=2`, `d_model=128`, `d_head=32`, `n_head=8`, `d_inner=256`, `n_head_softmax=8`, `d_head_softmax=16`, `dropout=0.05`, `conv_kernel_size=3`, `n_conv_layer=2`, `d_kernel_map=512`, `model_normalization=preLC`, `head_initialization=forward`, `softmax_attn=True`, `min_lr_ratio=0.004` and batch size 16.

| Checkpoint directory | Dataset | Input length | `pool_size` | `beta_hat_2` | Learning rate | `clip` | Checkpoints used |
|---|---|---|---|---|---|---|---|
| `ascadr` | ASCAD random key | 1,400 | 10 | 50 | 2.5e-4 | 0.25 | 1, 10, 20, ..., 90, 99 |
| `ches25_lx_w100`, `ches25_lx_w200` | CHES-CTF-2025 | 100, 200 | 5 | 150 | 2.5e-4 | 0.25 | 1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 201 |
| `ches25_lx_w500` | CHES-CTF-2025 | 500 | 10 | 150 | 2.5e-4 | 0.25 | same |
| `ches25_lx_w1000`, `ches25_lx_w3000`, `ches25_lx` | CHES-CTF-2025 | 1,000, 3,000, 7,000 | 20 | 150 | 2.5e-4 | 0.25 | same |
| `ascadf_desync50`, `ascadf_desync100` | ASCAD fixed key, desync 50 and 100 | 700 | 5 | 150 | 2.5e-4 | 0.25 | 60 and 40 (Figure 2) |
| `aes_rd_fstd_pool5_lr5e4` | AES-RD, feature-standardized | 3,500 | 5 | 150 | 5e-4 | 1.0 | 11 (Figure 2) |

- ASCAD and CHES-CTF-2025: `train_steps=4000000`, `warmup_steps=1000000`, `iterations=20000`.
- AES-RD: `train_steps=200000`, `warmup_steps=2000`, `iterations=2000`; input file from `bi_scope_certificates/prepare_aesrd_estranet_h5.py`.
- CHES-CTF-2025: `CHES25_PREPROCESS=lucianoxu`, `CHES25_LX_DENOISE=1`, `CHES25_LX_PROFILE_TRACES=450000`, `CHES25_LX_ATTACK_TRACES=-1`.
- Every run saves a checkpoint every 20,000 steps.
