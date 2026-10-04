---
title: "locality_diagnostics: how much of the trace a model needs"
---

# What this folder shows

This check from the long version takes a trained network and shows it only a short window of each trace, with everything outside the window set to zero. For each window length it finds a good window position and records how often the network still guesses right. The paper plots that success rate against the window length.

# Before you start

The Docker image (or the Python environment with `requirements-ml.txt`) and the TCHES20 download, which includes the ASCAD traces file and the trained networks. One window length with one seed takes about a minute. The full sweep in the paper (nine lengths, five seeds, a search over positions) takes about an hour.

# Run

```bash
T=datasets/raw/tches20/TCHES20V3_CNN_SCA
python -m locality_diagnostics.ascad_temporal_window_sweep \
  --ascad-h5 $T/datasets/ASCAD_dataset/ASCAD.h5 \
  --cnn-model $T/models/pretrained_models/models/zaid_ascad_desync_0_no_preprocessing_0.hdf5 \
  --tau-values 50,700 --n-seeds 1 --strategy center --output-dir results/locality_diagnostics
```

```{.text transcript="transcripts/08_locality_diagnostics/sweep.txt"}
```

```{.text transcript="transcripts/08_locality_diagnostics/outputs.txt"}
```

# What to look for

- One row per window length and seed: where the window was placed, the success rate inside the window, how many traces were tested, and the success rate on the full trace for comparison.
- The script mixes training and test traces when it builds its test set, so the success rate is not a clean out-of-sample number. The paper presents this figure as a diagnostic, not as a certificate.
