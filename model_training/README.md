# model_training

**What this folder does.** Small training examples for the three attacker families of the paper (MLP, CNN, Transformer). They are not the paper's reference models: the reported certificates use the pretrained TCHES20 Keras models and the EstraNet checkpoints, and nothing in the reproduction trains a model.

## Files

| File | Purpose | Framework |
|---|---|---|
| `train_mlp_example.py` | MLP profiled attacker | scikit-learn |
| `train_cnn_example.py` | 1D CNN profiled attacker | PyTorch |
| `train_transformer_example.py` | Transformer-style attacker for longer traces | PyTorch |
| `common.py` | shared loading, splitting, standardization and metadata helpers | |

## Before you run

- `requirements.txt` for the MLP; `requirements-ml.txt` for the CNN and the Transformer.
- Traces: `datasets/raw/ascad/ASCAD.h5` (`--dataset ascad`, see `datasets/README.md`), `datasets/raw/aes_hd/aes_hd_ext.npz` (`--dataset aes_hd`), or the TCHES20 checkout with the tags `ascad_desync_0`, `ascad_desync_50`, `ascad_desync_100`, `aes_rd`, `dpav4` (pass `--data-dir <checkout>/datasets --tches20-src-dir <checkout>/src`).
- Hardware: the MLP and the CNN at the default sizes run on CPU in minutes. The Transformer example uses a CUDA GPU when one is available; on CPU, reduce the number of epochs or the window length.

## Run

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

`--window-start` and `--window-end` select a sample range, which must lie within the trace length of the file (`ASCAD.h5` traces have 700 samples).

## Outputs

Checkpoints and `metadata.json` under `checkpoints/model_training/<dataset>/`, which git ignores: the MLP saves a scikit-learn pickle (`model.pkl`), the CNN and Transformer save a PyTorch state dict (`model.pt`) and a scaler pickle. These files are not loadable by the BI evaluators in `bi_suite_8datasets/` and `bi_scope_certificates/`, which load TCHES20 Keras `.hdf5` models or EstraNet TensorFlow checkpoints.

## Notes

Each example draws a random train/validation/holdout split (60/20/20 by default) from the traces returned by `core.data_loader.load_dataset`. For `--dataset ascad` that loader pools the ASCAD profiling and attack traces before the split, so attack traces can appear in the training set and the printed holdout accuracy is not an attack-partition result. For the TCHES20 tags the loader returns only the profiling traces.
