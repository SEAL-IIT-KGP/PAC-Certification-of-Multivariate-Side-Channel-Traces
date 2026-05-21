# Model Training Examples

This directory contains minimal local examples for the three attacker families used in the paper:

- `train_mlp_example.py`: MLP profiled attacker.
- `train_cnn_example.py`: 1D CNN profiled attacker.
- `train_transformer_example.py`: Transformer-style attacker for longer traces.
- `common.py`: shared dataset loading, splitting, standardization, and metadata helpers.

Example commands:

```bash
python -m model_training.train_mlp_example \
  --dataset ascad \
  --data-path datasets/raw/ascad/ASCAD.h5 \
  --window-start 45000 \
  --window-end 46500

python -m model_training.train_cnn_example \
  --dataset aes_hd \
  --data-path datasets/raw/aes_hd/aes_hd_ext.npz

python -m model_training.train_transformer_example \
  --dataset ascad_desync_100 \
  --data-dir datasets/raw/tches20/TCHES20V3_CNN_SCA/datasets \
  --tches20-src-dir datasets/raw/tches20/TCHES20V3_CNN_SCA/src
```

The examples write checkpoints and metadata under `checkpoints/model_training/`, which is ignored by git. Trained checkpoints can then be evaluated by the BI-suite and attacker-scope scripts.
