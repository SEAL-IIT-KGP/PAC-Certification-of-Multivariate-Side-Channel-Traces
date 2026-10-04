---
title: "attacker_scope_diagnostics: is the set of models big enough?"
---

# What this folder shows

A BI value is only as good as the set of models it covers. This folder trains sets of models of growing size and power on the same traces, and watches whether the certified success rate keeps growing. When adding stronger models no longer raises it, the set is large enough for that trace set.

- `attacker_scope_diagnostic.py`: the full run over trace sets, four model classes ("scopes") and seeds, with figures.
- `scope_cell_runner.py`: one trace set, one seed, one scope. This is how the run is split over a cluster.
- `aesrd_scope_seed_runner.py`: the same for AES-RD with a training budget.
- `evaluate_ascad_d0_challenger_fixed_split.py`: re-tests extra models on a fixed split of the ASCAD traces.

Scopes 1 and 2 use scikit-learn models. Scopes 3 and 4 add small PyTorch networks.

# Before you start

The Docker image (or the Python environment with `requirements-ml.txt`) and the TCHES20 download. These runs are slow: with all 700 points of each trace, one scope-1 cell trains six models with 256 classes each and takes hours on a CPU, which is why the paper ran them on a cluster. The example below keeps only the 50 most variable points (`--max-features 50`) so that it finishes in minutes.

# Run

One cell, with 50 trace points:

```bash
T=datasets/raw/tches20/TCHES20V3_CNN_SCA
python -m attacker_scope_diagnostics.scope_cell_runner --dataset ascad_desync_0 --seed 0 --scope 1 --max-features 50 \
  --dataset-dir $T/datasets --tches20-src-dir $T/src --device cpu \
  --output-dir results/attacker_scope_diagnostic --checkpoint results/attacker_scope_diagnostic/scope_checkpoint.csv
```

```{.text transcript="transcripts/09_attacker_scope_diagnostics/cell.txt"}
```

```{.text transcript="transcripts/09_attacker_scope_diagnostics/outputs.txt"}
```

The full run over scopes and seeds is `attacker_scope_diagnostics.attacker_scope_diagnostic` with `--datasets`, `--scopes` and `--n-seeds`; it uses all 700 points and takes hours per trace set.

# What to look for

- Each row of the results file gives the trace set, seed and scope, how many models were trained (`M`), the best success rate (`p_max`), and the BI in bits. In the 50-point example above, six linear models reach a best rate of 0.0046 and a BI of 3.06 bits. Reading the rows of one trace set from scope 1 to 4 shows whether the rate keeps rising.
- To run several trace sets at once, quote them: `--datasets "ascad_desync_0 aes_rd"`.
- The numbers in the paper always come from the fixed set of provided models (guides 03 and 04), never from the models trained here.
