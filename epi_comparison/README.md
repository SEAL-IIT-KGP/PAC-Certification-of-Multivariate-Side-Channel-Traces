# epi_comparison

**What this folder does.** It compares BI with effective perceived information (EPI, the power-recalibrated PI of Ito, Ueno and Homma) on the same frozen model, and produces the temperature-rescaling table of the full version (`tab:epi-temperature`, summarized in Section 5.2): rescaling the model's log-probabilities changes raw PI by tens of bits, leaves EPI unchanged, and changes no predicted label. Level D of `paper_results/` checks the table.

## Files

| File | Purpose |
|---|---|
| `epi_head_to_head.py` | the comparison on one TCHES20 benchmark |
| `summarize_epi_head_to_head.py` | verifies the checksums of the six result directories and writes the summary tables, including the temperature spans |

`epi_head_to_head.py` does five things: splits the profiling traces into three disjoint, class-balanced parts (selection, calibration, evaluation; seed 20260721); evaluates all 80 pretrained models on the selection part and freezes the most accurate one; fits the single inverse-temperature parameter of EPI on the calibration part and reports raw PI, EPI and the finite-benchmark BI of the frozen model on the evaluation part, with bootstrap intervals; rescales the log-probabilities by temperatures from 0.05 to 20 and recalibrates EPI at each; and repeats the calibration with fewer calibration traces.

## Before you run

`requirements-ml.txt` (TensorFlow) and the TCHES20 benchmark (`paper_results/fetch_tches20.sh`).

## Run

```bash
T=datasets/raw/tches20/TCHES20V3_CNN_SCA
for i in 0 1 2 3 4 5; do
  python -m epi_comparison.epi_head_to_head --dataset-index $i \
    --dataset-dir $T/datasets --tches20-src-dir $T/src --models-dir $T/models \
    --output-dir results/epi_comparison/main
done
python -m epi_comparison.summarize_epi_head_to_head --input-dir results/epi_comparison/main --output-dir results/epi_comparison/summary
```

`--dataset ascad_desync_0` (or any of the six tags) selects one benchmark by name instead of `--dataset-index`.

## Outputs

Per benchmark: `result.json`, `model_selection.csv`, `temperature_stress.csv` and `calibration_scarcity.csv`. The summary writes `summary.csv`, `temperature_all.csv`, `scarcity_all.csv` and `REPORT.md`.

## Notes

The EPI estimator (`fit_beta`, `information_records`) is the same as in `paper_results/matched_metrics.py`, which computes the EPI column of Table 7 on the attack partitions.
