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

The released checkpoints make retraining unnecessary. Retraining needs an NVIDIA GPU: a folder takes 4,000,000 steps at batch size 16 (AES-RD 200,000). On one NVIDIA RTX A4000, `ascadr` takes 8 h 41 min, `ches25_lx_w3000` 7 h 12 min, and `ascadf_desync50` and `ascadf_desync100` about 10 h 40 min each. [`docs/guides/pdf/12_gpu_docker.pdf`](../../docs/guides/pdf/12_gpu_docker.pdf) walks through it with recorded output and the time of each step.

**With Docker** (Linux x86-64, NVIDIA driver and NVIDIA Container Toolkit): build the GPU image and define a helper that shares the four folders and the GPU:

```bash
docker build --build-arg TF=gpu -t bi-certification-gpu .
mkdir -p results datasets/raw checkpoints external
gpu() {
  docker run --rm --gpus all -v "$PWD/results:/artifact/results" -v "$PWD/datasets/raw:/artifact/datasets/raw" \
    -v "$PWD/checkpoints:/artifact/checkpoints" -v "$PWD/external:/artifact/external" \
    bi-certification-gpu "$@"
}
```

```bash
# traces (CHES_Challenge.h5 is a manual download, see paper_results/README.md, Level C)
gpu bash -c "python -m paper_results.fetch_ascad_random_key && paper_results/fetch_tches20.sh \
  && python -m bi_scope_certificates.prepare_aesrd_estranet_h5"

# one folder
gpu bi_suite_8datasets/estranet_support/train_checkpoints.sh ascadr

# all ten, one after another
for suite in ascadr ascadf_desync50 ascadf_desync100 aes_rd_fstd_pool5_lr5e4 ches25_lx_w100 \
             ches25_lx_w200 ches25_lx_w500 ches25_lx_w1000 ches25_lx_w3000 ches25_lx; do
  gpu bi_suite_8datasets/estranet_support/train_checkpoints.sh "$suite"
done

# test the retrained Table 7 folders (ascadr, ches25_lx_w3000) in the CPU image (artifact() of the main README)
artifact python -m paper_results.evaluate_estranet_suites --checkpoint-root checkpoints/estranet_retrained \
  --out-dir results/paper_results/retrained
```

**Without Docker:** install `requirements-ml-gpu.txt` instead of `requirements-ml.txt` (Linux x86-64) and run `bi_suite_8datasets/estranet_support/train_checkpoints.sh <folder>` from the repository root.

| Folder | Trace file in `datasets/raw/` |
|---|---|
| `ascadr` | `ascad_random_key/ASCAD.h5` (`fetch_ascad_random_key`) |
| `ascadf_desync50`, `ascadf_desync100` | `tches20/.../ASCAD_desync50.h5`, `ASCAD_desync100.h5` (`fetch_tches20.sh`) |
| `aes_rd_fstd_pool5_lr5e4` | `aes_rd/AES_RD_estranet_feature_standardized.h5` (`prepare_aesrd_estranet_h5`) |
| `ches25_lx_w100`, `_w200`, `_w500`, `_w1000`, `_w3000`, `ches25_lx` | `ches_ctf_2025/CHES_Challenge.h5` |

- The script prints `gpus=1` when TensorFlow sees the GPU.
- Checkpoints go to `checkpoints/estranet_retrained/<folder>/trans_long-<n>`: `n = 1` is the untrained model, each later index adds 20,000 steps, `n = 201` ends the run. The same command resumes a stopped run.
- `--suites ches_windows` tests the window folders of Figure 1; for the Figure 2 reference models pass the same `--checkpoint-root` to `bi_scope_certificates/profile_epsilon_calibration.py`.

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
