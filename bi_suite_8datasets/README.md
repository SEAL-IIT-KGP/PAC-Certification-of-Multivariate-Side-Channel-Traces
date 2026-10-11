# bi_suite_8datasets

Every BI value starts from one integer per model: how many attack traces it classifies correctly. This folder produces those counts for the eight benchmarks of Table 7.

## Files

| File | Purpose |
|---|---|
| `evaluate_tches20_pretrained_fixed_split.py` | the 80 pretrained TCHES20 models of one benchmark (six rows of Table 7) |
| `evaluate_estranet_bi_suite.py` | a folder of EstraNet checkpoints (ASCAD random-key and CHES-CTF-2025 rows) |
| `summarize_ascad_estranet_bi_baselines.py`, `summarize_ches_transformer_bi_sweep.py` | turn EstraNet counts into certificate rows |
| `estranet_support/` | CHES-CTF-2025 loader, training patch and settings of the released checkpoints |

## Run: TCHES20 models

Needs `requirements-ml.txt` and the TCHES20 benchmark (`paper_results/fetch_tches20.sh`). Certificates use the attack partition:

```bash
T=datasets/raw/tches20/TCHES20V3_CNN_SCA
python -m bi_suite_8datasets.evaluate_tches20_pretrained_fixed_split \
  --dataset ascad_desync_0 --split attack \
  --models-dir $T/models/pretrained_models/models --dataset-dir $T/datasets --tches20-src-dir $T/src \
  --output-dir results/bi_suite_8datasets/tches20_fixed_split
```

For AES-RD, add `--max-eval-traces 12500` (the paper uses the first 12,500 attack traces). `--split profiling` and `--split both` are diagnostics on the training traces.

## Run: EstraNet checkpoints

Needs the EstraNet sources, checkpoints and traces of Level C (`paper_results/README.md`). The wrapper uses the paper's settings:

```bash
python -m paper_results.evaluate_estranet_suites --suites ches_ctf_2025 --skip-selected --out-dir results/bi_suite_8datasets/estranet
```

## Outputs

| File | Content |
|---|---|
| `<dataset>_<split>_success.csv` | one row per model: successes and number of traces |
| `<dataset>_<split>_summary.csv` | the suite certificate: `p_obs`, KL-binomial endpoints at `delta = 1e-6` with the union bound over `M` models, and bits |
| EstraNet: `--output-csv` and `<stem>.summary.json` | one row per checkpoint |
