# bi_suite_8datasets

**What this folder does.** Every BI value starts from one integer per model: how many attack traces it classifies correctly (its top guess equals the true S-box output byte). This folder produces those counts for the eight benchmarks of Table 7. Key-rank curves are separate evidence and are not used as BI input.

## Files

| File | Purpose |
|---|---|
| `evaluate_tches20_pretrained_fixed_split.py` | all 80 pretrained TCHES20 Keras models of one benchmark on one split; the six TCHES20 rows of Table 7 |
| `evaluate_estranet_bi_suite.py` | a folder of EstraNet Transformer checkpoints; the ASCAD random-key and CHES-CTF-2025 rows |
| `summarize_ascad_estranet_bi_baselines.py`, `summarize_ches_transformer_bi_sweep.py` | turn EstraNet counts into certificate rows |
| `estranet_support/` | the CHES-CTF-2025 loader added to the EstraNet sources, the training patch, and the settings of every released checkpoint (see its README) |

`paper_results/evaluate_estranet_suites.py` runs the EstraNet evaluator with the paper's settings for both Table 7 suites and for the CHES-CTF-2025 window suites of Figure 1.

## Before you run

- `requirements-ml.txt` (TensorFlow).
- The TCHES20 benchmark: `paper_results/fetch_tches20.sh` clones it into `datasets/raw/tches20/TCHES20V3_CNN_SCA` and unpacks the split model archive into `models/pretrained_models/models`.
- For EstraNet: the sources (`paper_results/fetch_estranet.sh`), the checkpoint archive, and the trace files (see `paper_results/README.md`, Level C).

## Run: TCHES20 models

Certificates use the official attack partition, so always pass `--split attack`:

```bash
T=datasets/raw/tches20/TCHES20V3_CNN_SCA
python -m bi_suite_8datasets.evaluate_tches20_pretrained_fixed_split \
  --dataset ascad_desync_0 --split attack \
  --models-dir $T/models/pretrained_models/models --dataset-dir $T/datasets --tches20-src-dir $T/src \
  --output-dir results/bi_suite_8datasets/tches20_fixed_split
```

For AES-RD the paper uses the first 12,500 attack records: add `--dataset aes_rd --max-eval-traces 12500`.

## Run: EstraNet checkpoints

```bash
python -m bi_suite_8datasets.evaluate_estranet_bi_suite --help
python -m paper_results.evaluate_estranet_suites --suites ches_ctf_2025 --skip-selected --out-dir results/bi_suite_8datasets/estranet
```

## Outputs

| File | Content |
|---|---|
| `<dataset>_<split>_success.csv` | one row per model: successes and number of traces |
| `<dataset>_<split>_summary.csv` | the suite certificate: best rate `p_obs`, KL-binomial endpoints at `delta = 1e-6` with the union bound over `M = 80` models, and bits |
| EstraNet: the `--output-csv` file and `<stem>.summary.json` | one row per checkpoint |

## Notes

- `--split profiling` and `--split both` evaluate the models on their own training traces. They are in-sample diagnostics, not certificates; the script prints a warning.
- Near-chance EstraNet checkpoints (about 390 of 100,000) can differ by one trace between machines. The suite maximum does not.
- Write outputs under `results/` or another ignored path.
