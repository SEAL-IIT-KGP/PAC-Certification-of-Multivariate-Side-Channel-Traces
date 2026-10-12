---
title: "GPU Docker: training the Transformer models again"
---

# What this guide shows

Every other guide runs on an ordinary CPU, because the trained models are provided. This guide is for one optional task: training the Transformer models again from scratch on an NVIDIA GPU, then testing the new models the same way as the provided ones. It uses a second Docker image, `bi-certification-gpu`, built from the same Dockerfile with one extra option.

What the GPU image can train again:

| Results | Models behind them | Trained again here |
|------------------------------------|--------------------------------|-------------|
| Table 7, ASCAD random-key and CHES-CTF-2025 rows; Table 8, ASCAD random-key row | Transformer folders `ascadr` and `ches25_lx_w3000` | yes |
| Figure 1, BI line | the same CHES-CTF-2025 folder and `ches25_lx_w100`, `_w200`, `_w500`, `_w1000`, `ches25_lx` | yes |
| Figure 2, four Transformer reference models | `ascadr` and `ascadf_desync50`, `ascadf_desync100`, `aes_rd_fstd_pool5_lr5e4` | yes |
| Figure 2, two Keras reference models | Keras MLP and CNN of ASCAD random key | yes, on CPU (guide 04) |
| Table 7, six TCHES20 rows | 480 public models of the TCHES20 benchmark | no: they are downloaded, as in Level B |

Training runs in the GPU image. The new models are then tested in the ordinary CPU image `bi-certification`, exactly like the provided models in Level C.

# What you need

- A Linux x86-64 machine with an NVIDIA GPU, the NVIDIA driver and the NVIDIA Container Toolkit. The outputs below were recorded on a Linux workstation with one NVIDIA RTX A4000.
- About 6 GB of disk for the image, plus the trace files of the folders you want to train (table in Step 3).

Check the driver and the toolkit on the host:

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv
nvidia-ctk --version
```

```{.text transcript="transcripts/12_gpu_docker/nvidia_smi.txt"}
```

# Step 1: build the GPU image

From the top folder of the repository:

```bash
docker build --build-arg TF=gpu -t bi-certification-gpu .
```

```{.text transcript="transcripts/12_gpu_docker/docker_build_gpu.txt"}
```

The option `TF=gpu` installs `requirements-ml-gpu.txt`: TensorFlow 2.15 with its CUDA 12.2 and cuDNN 8.9 libraries. Everything else is the same as in `bi-certification`. Without the option you get the ordinary CPU image.

# Step 2: open a shell in the GPU image

The GPU image shares the same four folders as the CPU image (guide 00). Add `--gpus all` so the container can use the GPU:

```bash
mkdir -p results datasets/raw checkpoints external
docker run -it --rm --gpus all \
  -v "$PWD/results:/artifact/results" \
  -v "$PWD/datasets/raw:/artifact/datasets/raw" \
  -v "$PWD/checkpoints:/artifact/checkpoints" \
  -v "$PWD/external:/artifact/external" \
  bi-certification-gpu bash
```

Type the commands of the next steps at that prompt. First check that TensorFlow sees the GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

```{.text transcript="transcripts/12_gpu_docker/gpu_visible.txt"}
```

At the start of every GPU command, TensorFlow prints three lines `Unable to register cuDNN/cuFFT/cuBLAS factory`. They are harmless.

# Step 3: get the traces

Each Transformer folder is trained on one trace file:

| Folders | Trace file in `datasets/raw/` | How to get it |
|----------------------------|-------------------------------------|----------------------------|
| `ascadr` | `ascad_random_key/ASCAD.h5` | `python -m paper_results.fetch_ascad_random_key` |
| `ascadf_desync50`, `ascadf_desync100` | `tches20/.../ASCAD_desync50.h5`, `ASCAD_desync100.h5` | `paper_results/fetch_tches20.sh` |
| `aes_rd_fstd_pool5_lr5e4` | `aes_rd/AES_RD_estranet_feature_standardized.h5` | `fetch_tches20.sh`, then `python -m bi_scope_certificates.prepare_aesrd_estranet_h5` |
| the six `ches25_lx…` folders | `ches_ctf_2025/CHES_Challenge.h5` | download by hand (guide 00) |

The three scripts in one command:

```bash
python -m paper_results.fetch_ascad_random_key && paper_results/fetch_tches20.sh && \
  python -m bi_scope_certificates.prepare_aesrd_estranet_h5
```

```{.text transcript="transcripts/12_gpu_docker/fetch_traces.txt"}
```

# Step 4: train one Transformer folder

`train_checkpoints.sh` trains one folder with the settings of the paper (README of `bi_suite_8datasets/estranet_support/`). It downloads the Transformer code into `external/` if it is missing, prints the settings and the number of GPUs it sees, and saves a model every 20,000 steps:

```bash
bi_suite_8datasets/estranet_support/train_checkpoints.sh ascadr
```

```{.text transcript="transcripts/12_gpu_docker/train_full_ascadr.txt"}
```

- The new models go to `checkpoints/estranet_retrained/<folder>/trans_long-<n>`, next to the provided ones in `checkpoints/estranet/`. Model 1 is the untrained model, and model `n` has trained for 20,000 × (n − 1) steps; model 201 ends the run.
- If a run stops, the same command continues from the last saved model.
- `gpus=0` at the end of the `train_steps=` line means the container does not see the GPU (check `--gpus all` and Step 2). Training would then run on the CPU, which takes days per folder.

To train all ten folders, one after another, run from the host:

```bash
for suite in ascadr ascadf_desync50 ascadf_desync100 aes_rd_fstd_pool5_lr5e4 ches25_lx_w100 \
             ches25_lx_w200 ches25_lx_w500 ches25_lx_w1000 ches25_lx_w3000 ches25_lx; do
  docker run --rm --gpus all -v "$PWD/results:/artifact/results" -v "$PWD/datasets/raw:/artifact/datasets/raw" \
    -v "$PWD/checkpoints:/artifact/checkpoints" -v "$PWD/external:/artifact/external" \
    bi-certification-gpu bi_suite_8datasets/estranet_support/train_checkpoints.sh "$suite"
done
```

# Step 5: test the new models

Leave the GPU shell and open the CPU shell of guide 00 (image `bi-certification`, the same four shared folders). The Level C script takes the folder of the new models with `--checkpoint-root`. For the two Table 7 folders:

```bash
python -m paper_results.evaluate_estranet_suites --suites ascad_random_key \
  --checkpoint-root checkpoints/estranet_retrained --out-dir results/paper_results/retrained
python -m paper_results.evaluate_estranet_suites --suites ches_ctf_2025 \
  --checkpoint-root checkpoints/estranet_retrained --out-dir results/paper_results/retrained
```

```{.text transcript="transcripts/12_gpu_docker/eval_retrained_ascad_random_key.txt"}
```

```{.text transcript="transcripts/12_gpu_docker/eval_retrained_ches_ctf_2025.txt"}
```

Each test prints the correct guesses of the eleven models of the folder, then EPI and key recovery of the model with the most correct guesses, as in Level C. Table 7 from these counts:

```bash
python -m paper_results.table7_bi_suite --recomputed-dir results/paper_results/retrained \
  --out-dir results/paper_results/retrained_table7
cat results/paper_results/retrained_table7/table7_bi_suite.md
```

```{.text transcript="transcripts/12_gpu_docker/table7_retrained.txt"}
```

The two rows marked `recomputed` come from the new models; the six TCHES20 rows keep their archived counts. For ASCAD random key the new models give a certified bound BI+ of 2.301 bits (2.313 with the provided models), and for CHES-CTF-2025 0.509 bits (0.521).

For the window folders of Figure 1 add `--suites ches_windows`. For the Figure 2 reference models, pass the same `--checkpoint-root` to `bi_scope_certificates.profile_epsilon_calibration`.

# How long each step takes

Measured on the Linux workstation with one NVIDIA RTX A4000. A full run is 4,000,000 training steps (AES-RD 200,000).

| Step | Image | Time |
|------------------------------------------|---------|-------------|
| build the GPU image (first time) | GPU | 1.5 minutes |
| get the traces (Step 3) | GPU | 30 seconds, plus the downloads |
| train `ascadr` (Table 7, Table 8, Figure 2) | GPU | 8 hours 41 minutes |
| train `ches25_lx_w3000` (Table 7, Figure 1) | GPU | 7 hours 12 minutes |
| train `ascadf_desync50` (Figure 2) | GPU | about 10 hours 40 minutes |
| train `ascadf_desync100` (Figure 2) | GPU | about 10 hours 40 minutes |
| test the new `ascadr` models | CPU | 9.5 minutes |
| test the new `ches25_lx_w3000` models | CPU | 6.5 minutes |
| Table 7 from the new counts | CPU | 1 second |

The two "about" times come from the speed over 40,000 steps; for `ascadr` the same method gives 8 hours 38 minutes, against 8 hours 41 minutes for the full run. The times of the remaining six folders will be added here.

# What to look for

- `gpus=1` at the end of the `train_steps=` line of every training run.
- Twenty more saved models per 400,000 steps in `checkpoints/estranet_retrained/<folder>/`.
- The test of the new models prints `successes=…` for each of the eleven models of a folder, and the selected model's EPI and key recovery, as in Level C.
