---
title: "model_training: small training examples"
---

# What this folder shows

The paper's results use the provided trained models (the published TCHES20 models and the released Transformer checkpoints), so nothing in the reproduction trains a model. This folder shows, at a small scale, how each kind of model in the paper is trained: `train_mlp_example.py` (a plain network, scikit-learn), `train_cnn_example.py` (a convolutional network, PyTorch) and `train_transformer_example.py` (a small Transformer, PyTorch). The models they save cannot be used by the certificate scripts, which expect the provided model files.

# Before you start

`requirements.txt` for the plain network, `requirements-ml.txt` for the other two, and some traces: the TCHES20 download works with `--dataset ascad_desync_0`, `ascad_desync_50`, `ascad_desync_100`, `aes_rd` or `dpav4`. The first two examples run on a CPU in minutes. The Transformer example uses a GPU if one is present; on a CPU, lower the number of epochs or shorten the window.

# Run

```bash
T=datasets/raw/tches20/TCHES20V3_CNN_SCA
python -m model_training.train_mlp_example --dataset ascad_desync_0 --data-dir $T/datasets --tches20-src-dir $T/src
```

```{.text transcript="transcripts/10_model_training/mlp.txt"}
```

```bash
python -m model_training.train_cnn_example --dataset ascad_desync_0 --data-dir $T/datasets --tches20-src-dir $T/src --epochs 3
```

```{.text transcript="transcripts/10_model_training/cnn.txt"}
```

```{.text transcript="transcripts/10_model_training/outputs.txt"}
```

# What to look for

Each example prints the sizes of the training, validation and test parts, the training progress, and the accuracy on the test part. It saves the model and a `metadata.json` with the settings and results.
