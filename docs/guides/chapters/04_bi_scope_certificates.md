---
title: "bi_scope_certificates: covering models that are nearly the same"
---

# What this folder shows

A BI value for a set of models covers only those models. The paper extends it to every model whose scores stay within a small distance of one model in the set. The idea is simple: if the winning score of a model leads the runner-up by more than that distance, any nearby model makes the same guess. So the certificate is again a count: the traces where the model's lead is large enough.

| Script | What it produces | Used for |
|----------------------------------|--------------------------------------------|--------------------|
| `compute_full_suite_bi_loc.py` | these counts for all 80 models of one trace set, at nine distances | Table 9 |
| `profile_epsilon_calibration.py` | the counts on two parts of the traces, plus how much the scores move when the model is run again, in another batch size, in lower precision, or with noise | Figure 2 |
| `compute_bi_loc_margin_curves.py` | the same curves for chosen models on all their attack traces | numbers in Section 5.3 |
| `prepare_aesrd_estranet_h5.py` | the input file for the AES-RD Transformer | Level C |
| `build_bi_scope_tables.py` | tables from the curves | long version |

# Before you start

The Docker image (or the Python environment with `requirements-ml.txt`), the TCHES20 download, and the file `results/paper_results/bi_suite_dataset_summary.csv`, which Level A or B writes. The Transformer models also need the source code and models from guide 01, Level C.

# Run

Counts for the 80 ASCAD (desync 50) models at nine distances:

```bash
python -m bi_scope_certificates.compute_full_suite_bi_loc --dataset ascad_desync_50 \
  --suite-csv results/paper_results/bi_suite_dataset_summary.csv \
  --output-dir results/bi_scope_certificates/full_suite_loc
```

```{.text transcript="transcripts/04_bi_scope_certificates/full_suite.txt"}
```

```{.text transcript="transcripts/04_bi_scope_certificates/full_suite_summary.txt"}
```

The two public models of the same trace set, with the traces split 20% for choosing the distance and 80% for the certificate:

```bash
python -m bi_scope_certificates.profile_epsilon_calibration --dataset ascad_desync_50 --public-only \
  --output-dir results/bi_scope_certificates/calibration/ascad_desync_50
```

```{.text transcript="transcripts/04_bi_scope_certificates/profile.txt"}
```

```{.text transcript="transcripts/04_bi_scope_certificates/profile_outputs.txt"}
```

# What to look for

- The summary has one row per distance: the largest count over the 80 models, the matching rate, the limit in bits, and which model gave it. At distance 0 the count is the plain number of correct guesses. At the largest distance it covers almost every trace, which is why the table in the paper stops there.
- `epsilon_margin_counts.csv` holds the counts for both parts of the traces. The Figure 2 script picks the smallest distance that changes the count of at least 75% of the twelve models and is larger than the score movement seen when the models are rerun.
- `epsilon_variant_distances.csv` shows that movement.
