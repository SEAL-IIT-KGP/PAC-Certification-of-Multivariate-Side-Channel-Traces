# EstraNet Support Files

The Transformer attackers in the paper are EstraNet models
(https://github.com/suvadeep-iitb/EstraNet, commit `7f969ce`). The evaluators
import the upstream `transformer.py`, `fast_attention.py`, `normalization.py`,
and `data_utils.py` unchanged. This directory holds the files added on top of
the upstream sources:

- `data_utils_ches25.py`: loader for the CHES-CTF-2025 file `CHES_Challenge.h5`.
  It follows the layout of the upstream `data_utils.py`, targets byte 0 of the
  AES S-box output, and applies the preprocessing used for the CHES-CTF-2025
  suites: notch denoising, a shift of the attack-trace mean toward the
  profiling traces with key byte 127, and per-sample standardization fitted on
  the profiling traces (the scheme of https://github.com/LucianoXu/CHES2025-Challenge).
  The mean shift uses the mean of the unlabeled attack traces, which is why the
  paper reports the CHES-CTF-2025 row as a numerical diagnostic.
- `evaluation_utils_ches25.py`: key-rank evaluation for the same dataset, used
  only by the upstream training script.
- `train_trans.patch`: changes to the upstream `train_trans.py` used for
  training (adds the `CHES25` dataset option, a `rank_repeats` flag, and builds
  the model variables before a checkpoint is restored). Evaluation does not
  need this patch.

`paper_results/fetch_estranet.sh` clones the upstream repository at the commit
above into `external/EstraNet` and copies the two Python files into it. To
retrain, also apply the patch:

```bash
git -C external/EstraNet apply "$PWD/bi_suite_8datasets/estranet_support/train_trans.patch"
```

## Training settings of the released checkpoints

Training needs a CUDA-capable GPU: the ASCAD and CHES-CTF-2025 runs take
4,000,000 steps at batch size 16, which is impractical on CPU. Evaluating the
released checkpoints (Levels C and D of `paper_results/`) runs on CPU.

All runs use `train_trans.py` with `n_layer=2`, `d_model=128`, `d_head=32`,
`n_head=8`, `d_inner=256`, `n_head_softmax=8`, `d_head_softmax=16`,
`dropout=0.05`, `conv_kernel_size=3`, `n_conv_layer=2`, `d_kernel_map=512`,
`model_normalization=preLC`, `head_initialization=forward`,
`softmax_attn=True`, `min_lr_ratio=0.004`, and batch size 16.

| Checkpoint directory | Dataset | Input length | `pool_size` | `beta_hat_2` | Learning rate | `clip` | Checkpoints used in the paper |
|---|---|---|---|---|---|---|---|
| `ascadr` | ASCAD random key | 1,400 | 10 | 50 | 2.5e-4 | 0.25 | 1, 10, 20, ..., 90, 99 (suite) |
| `ches25_lx_w100`, `ches25_lx_w200` | CHES-CTF-2025 | 100, 200 | 5 | 150 | 2.5e-4 | 0.25 | 1, 2, 3, 4, 5, 10, 20, 50, 100, 150, 201 (suite) |
| `ches25_lx_w500` | CHES-CTF-2025 | 500 | 10 | 150 | 2.5e-4 | 0.25 | same |
| `ches25_lx_w1000`, `ches25_lx_w3000`, `ches25_lx` | CHES-CTF-2025 | 1,000, 3,000, 7,000 | 20 | 150 | 2.5e-4 | 0.25 | same |
| `ascadf_desync50`, `ascadf_desync100` | ASCAD fixed key (desync 50, 100) | 700 | 5 | 150 | 2.5e-4 | 0.25 | 60 and 40 (Figure 2 reference models) |
| `aes_rd_fstd_pool5_lr5e4` | AES-RD, feature-standardized | 3,500 | 5 | 150 | 5e-4 | 1.0 | 11 (Figure 2 reference model) |

The ASCAD and CHES-CTF-2025 launch scripts set `train_steps=4000000` and
`warmup_steps=1000000`; the AES-RD run sets `train_steps=200000` and
`warmup_steps=2000`. The CHES-CTF-2025 runs set `CHES25_PREPROCESS=lucianoxu`,
`CHES25_LX_DENOISE=1`, `CHES25_LX_PROFILE_TRACES=450000`, and
`CHES25_LX_ATTACK_TRACES=-1`. The AES-RD input file is produced by
`bi_scope_certificates/prepare_aesrd_estranet_h5.py`.
