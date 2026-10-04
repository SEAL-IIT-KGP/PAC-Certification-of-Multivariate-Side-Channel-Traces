---
title: "bi_suite_8datasets: counting correct guesses"
---

# What this folder shows

Every BI value starts from one number per model: how many of the attack traces it guessed correctly. This folder produces that number for the eight sets of traces in Table 7.

- `evaluate_tches20_pretrained_fixed_split.py`: runs all 80 trained TCHES20 models of one trace set and counts their correct guesses. This gives the six TCHES20 rows of Table 7.
- `evaluate_estranet_bi_suite.py`: does the same for a folder of saved Transformer models. This gives the ASCAD random-key and CHES-CTF-2025 rows. `paper_results/evaluate_estranet_suites.py` calls it with the settings used in the paper.
- `summarize_*.py`: turn the counts into table rows.
- `estranet_support/`: the loader for the CHES-CTF-2025 traces, the small patch used for training, and the training settings of every saved model.

A guess is correct when the model's top choice equals the true value of the key byte. Only this count feeds into BI.

# Before you start

The Docker image (or the Python environment with `requirements-ml.txt`) and the TCHES20 download (`paper_results/fetch_tches20.sh`). For the Transformer part, also the source code, the models and the traces from guide 01, Level C.

# Run: one set of TCHES20 models

Certificates are always computed on the attack traces, so pass `--split attack`:

```bash
T=datasets/raw/tches20/TCHES20V3_CNN_SCA
python -m bi_suite_8datasets.evaluate_tches20_pretrained_fixed_split \
  --dataset ascad_desync_0 --split attack \
  --models-dir $T/models/pretrained_models/models --dataset-dir $T/datasets --tches20-src-dir $T/src \
  --output-dir results/bi_suite_8datasets/tches20_fixed_split
```

```{.text transcript="transcripts/03_bi_suite_8datasets/eval_ascad_d0.txt"}
```

```{.text transcript="transcripts/03_bi_suite_8datasets/outputs.txt"}
```

For AES-RD add `--max-eval-traces 12500`, because the paper uses only the first 12,500 attack traces of that set.

# Run: a set of Transformer models

The Transformer script takes the model settings on the command line:

```bash
python -m bi_suite_8datasets.evaluate_estranet_bi_suite --help
```

```{.text transcript="transcripts/03_bi_suite_8datasets/estranet_help.txt"}
```

The wrapper below fills in the paper's settings. It tests the eleven saved CHES-CTF-2025 models on 100,000 traces:

```bash
python -m paper_results.evaluate_estranet_suites --suites ches_ctf_2025 --skip-selected \
  --out-dir results/bi_suite_8datasets/estranet
```

```{.text transcript="transcripts/03_bi_suite_8datasets/estranet_suite.txt"}
```

# What to look for

- `<dataset>_attack_success.csv`: one row per model with its number of correct guesses. The largest number is what BI is computed from.
- `<dataset>_attack_summary.csv`: the BI of the whole set of models: the best success rate, its guaranteed upper limit, and both in bits.
- The Transformer script prints `successes=… n_te=100000` for each model. Models that guess about as well as chance (around 390 out of 100,000) can differ by one trace from machine to machine. The best model does not.
- `--split profiling` tests the models on the traces they were trained on. That is only a sanity check, not a certificate, and the script says so.
