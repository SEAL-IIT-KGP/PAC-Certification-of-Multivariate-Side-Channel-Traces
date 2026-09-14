# Model Training Examples

This directory contains small illustrative training examples for the three attacker families discussed in the paper (MLP, CNN, and Transformer). They are not the paper's reference models: the reported certificates use the pretrained TCHES20 Keras models and the EstraNet checkpoints.

- `train_mlp_example.py`: MLP profiled attacker.
- `train_cnn_example.py`: 1D CNN profiled attacker.
- `train_transformer_example.py`: Transformer-style attacker for longer traces.
- `common.py`: shared dataset loading, splitting, standardization, and metadata helpers.

Example commands:

```bash
python -m model_training.train_mlp_example \
  --dataset ascad \
  --data-path datasets/raw/ascad/ASCAD.h5

python -m model_training.train_cnn_example \
  --dataset aes_hd \
  --data-path datasets/raw/aes_hd/aes_hd_ext.npz

python -m model_training.train_transformer_example \
  --dataset ascad_desync_100 \
  --data-dir datasets/raw/tches20/TCHES20V3_CNN_SCA/datasets \
  --tches20-src-dir datasets/raw/tches20/TCHES20V3_CNN_SCA/src
```

The optional `--window-start` and `--window-end` flags select a sample range, which must lie within the trace length of the loaded file. For example, `ASCAD.h5` traces have 700 samples.

The examples write checkpoints and metadata under `checkpoints/model_training/`, which is ignored by git. The MLP example saves a scikit-learn pickle (`model.pkl`). The CNN and Transformer examples save a PyTorch state dict (`model.pt`) and a scaler pickle. These files are not directly loadable by the BI evaluators in `bi_suite_8datasets/` and `bi_scope_certificates/`, which load TCHES20 Keras `.hdf5` models or EstraNet TensorFlow checkpoints.

Data split caveat: each example draws a random train/validation/holdout split (60/20/20 by default) from the traces returned by `core.data_loader.load_dataset`. For `--dataset ascad`, that loader pools the ASCAD profiling and attack traces before the split, so attack traces can appear in the training set. The printed holdout accuracy is therefore not an attack-partition result. For the TCHES20 dataset tags (`ascad_desync_0`, `ascad_desync_50`, `ascad_desync_100`, `aes_rd`, `dpav4`), the loader returns only the profiling traces.
