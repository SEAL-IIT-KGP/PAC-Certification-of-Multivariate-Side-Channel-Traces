# epi_comparison

Compares BI with effective perceived information (EPI, Ito, Ueno and Homma) on the same frozen model. It produces the temperature-rescaling table of the full version (summarized in Section 5.4): rescaling the model's log-probabilities changes raw PI by tens of bits, leaves EPI unchanged, and changes no predicted label. Level D of `paper_results/` checks this table.

## Files

| File | Purpose |
|---|---|
| `epi_head_to_head.py` | the comparison on one TCHES20 benchmark |
| `summarize_epi_head_to_head.py` | verifies the six result folders and writes the summary tables |

For one benchmark, `epi_head_to_head.py`:

1. splits the profiling traces into selection, calibration and evaluation parts (class-balanced, seed 20260721);
2. freezes the most accurate of the 80 pretrained models on the selection part;
3. fits EPI on the calibration part and reports raw PI, EPI and BI on the evaluation part, with bootstrap intervals;
4. repeats EPI at nine temperatures from 0.05 to 20, and with fewer calibration traces.

## Run

Needs `requirements-ml.txt` and the TCHES20 benchmark (`paper_results/fetch_tches20.sh`):

```bash
T=datasets/raw/tches20/TCHES20V3_CNN_SCA
for i in 0 1 2 3 4 5; do
  python -m epi_comparison.epi_head_to_head --dataset-index $i \
    --dataset-dir $T/datasets --tches20-src-dir $T/src --models-dir $T/models \
    --output-dir results/epi_comparison/main
done
python -m epi_comparison.summarize_epi_head_to_head --input-dir results/epi_comparison/main --output-dir results/epi_comparison/summary
```

`--dataset ascad_desync_0` (or any of the six tags) selects a benchmark by name.

## Outputs

- Per benchmark: `result.json`, `model_selection.csv`, `temperature_stress.csv`, `calibration_scarcity.csv`.
- Summary: `summary.csv`, `temperature_all.csv`, `scarcity_all.csv`, `REPORT.md`.

The EPI estimator is the same as in `paper_results/matched_metrics.py`, which computes the EPI column of Table 7.
