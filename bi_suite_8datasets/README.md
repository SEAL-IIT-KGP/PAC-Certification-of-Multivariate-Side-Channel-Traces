# BI Suite Evaluation

This directory contains evaluators for BI-suite rows that are computed from declared checkpoint suites:

- `ascad_random_key`: EstraNet checkpoint suite from `checkpoints/ascadr`
- `ches_ctf_2025`: EstraNet 3000-window checkpoint suite from `checkpoints/ches25_lx_w3000`

The evaluator computes single-target classification success on the attack split for every declared checkpoint. These success counts are the inputs to the KL-binomial `BI^{suite}` bracket. Key-rank and GE curves remain separate SCSCA/SASCA evidence and are not used as BI input.

Main Python entry points:

- `evaluate_estranet_bi_suite.py`: evaluate EstraNet checkpoint suites.
- `evaluate_tches20_pretrained_fixed_split.py`: evaluate TCHES20-style pretrained fixed-split suites.
- `summarize_ascad_estranet_bi_baselines.py`: summarize ASCAD EstraNet BI rows.
- `summarize_ches_transformer_bi_sweep.py`: summarize CHES Transformer BI rows.

## TCHES20 pretrained suites

`evaluate_tches20_pretrained_fixed_split.py` evaluates every pretrained `.hdf5` model for one dataset on one split and writes `<dataset>_<split>_success.csv` and `<dataset>_<split>_summary.csv`. Certificates use the official attack partition, so pass `--split attack`.

The pretrained models ship in https://github.com/KULeuven-COSIC/TCHES20V3_CNN_SCA as a split archive (`models/pretrained_models.zip` and `models/pretrained_models.z01`). Extract it first and point `--models-dir` at the directory that contains the `.hdf5` files; the examples below assume it was extracted to `models/pretrained_models/models` inside the TCHES20 checkout.

```bash
python -m bi_suite_8datasets.evaluate_tches20_pretrained_fixed_split \
  --dataset ascad_desync_0 \
  --split attack \
  --models-dir datasets/raw/tches20/TCHES20V3_CNN_SCA/models/pretrained_models/models \
  --dataset-dir datasets/raw/tches20/TCHES20V3_CNN_SCA/datasets \
  --tches20-src-dir datasets/raw/tches20/TCHES20V3_CNN_SCA/src \
  --output-dir results/bi_margin_reduction_audit/tches20_fixed_split
```

For AES-RD, the paper uses the first 12,500 attack records. Add `--max-eval-traces 12500`:

```bash
python -m bi_suite_8datasets.evaluate_tches20_pretrained_fixed_split \
  --dataset aes_rd \
  --split attack \
  --max-eval-traces 12500 \
  --models-dir datasets/raw/tches20/TCHES20V3_CNN_SCA/models/pretrained_models/models \
  --dataset-dir datasets/raw/tches20/TCHES20V3_CNN_SCA/datasets \
  --tches20-src-dir datasets/raw/tches20/TCHES20V3_CNN_SCA/src \
  --output-dir results/bi_margin_reduction_audit/tches20_fixed_split
```

`--split profiling` and `--split both` evaluate the pretrained models on their own training (profiling) traces. They are in-sample diagnostics only, not valid certificates, and the script prints a warning when either is used.

Generated outputs should be written under a local `results/` directory or another ignored output path.
