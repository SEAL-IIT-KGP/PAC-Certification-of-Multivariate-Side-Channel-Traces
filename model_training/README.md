# model_training

Small training examples for the three attacker families of the paper (MLP, CNN, Transformer). The paper's results use the pretrained TCHES20 models and the released checkpoints; to retrain those, see `bi_suite_8datasets/estranet_support/` and `bi_scope_certificates/train_keras_ascad_random_key.py`.

## Files

| File | Purpose | Framework |
|---|---|---|
| `train_mlp_example.py` | MLP attacker | scikit-learn |
| `train_cnn_example.py` | 1D CNN attacker | PyTorch |
| `train_transformer_example.py` | Transformer-style attacker for longer traces | PyTorch |
| `common.py` | shared loading, splitting and metadata helpers | |

## Run

Needs `requirements.txt` (MLP) or `requirements-ml.txt` (CNN, Transformer). The MLP and CNN run on CPU in minutes; the Transformer uses a GPU when available.

```bash
python -m model_training.train_mlp_example --dataset ascad --data-path datasets/raw/ascad/ASCAD.h5

python -m model_training.train_cnn_example --dataset aes_hd --data-path datasets/raw/aes_hd/aes_hd_ext.npz

python -m model_training.train_transformer_example --dataset ascad_desync_100 \
  --data-dir datasets/raw/tches20/TCHES20V3_CNN_SCA/datasets \
  --tches20-src-dir datasets/raw/tches20/TCHES20V3_CNN_SCA/src
```

- Datasets: `ascad`, `aes_hd` (see `datasets/README.md`), or the TCHES20 tags `ascad_desync_0`, `ascad_desync_50`, `ascad_desync_100`, `aes_rd`, `dpav4`.
- `--window-start` and `--window-end` select a sample range (ASCAD traces have 700 samples).
- Each example splits the traces 60/20/20 into train, validation and holdout.

## Outputs

`checkpoints/model_training/<dataset>/`: the model (`model.pkl` for the MLP, `model.pt` and a scaler for the CNN and Transformer) and `metadata.json`.
