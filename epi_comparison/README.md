# Same-Model BI versus Effective PI

Code behind the full version's comparison of BI with effective perceived
information (EPI, the power-recalibrated PI of Ito, Ueno, and Homma) on the
same frozen model, including the temperature-rescaling table
(`tab:epi-temperature`) summarized in Section 5.2 of the paper.

`epi_head_to_head.py` works on one TCHES20 benchmark:

1. splits the profiling traces into three disjoint, class-balanced parts
   (selection, calibration, evaluation; seed 20260721);
2. evaluates all 80 pretrained models on the selection part and freezes the
   most accurate one;
3. fits the single inverse-temperature parameter of EPI on the calibration
   part and reports raw PI, EPI, and the finite-benchmark BI of the frozen model
   on the evaluation part, with bootstrap intervals;
4. rescales the model's log-probabilities by temperatures from 0.05 to 20,
   recalibrates EPI at each temperature, and records raw PI, EPI, and whether
   every predicted label is preserved (the temperature table);
5. repeats the calibration with fewer calibration traces (calibration scarcity).

`summarize_epi_head_to_head.py` verifies the checksums of the six result
directories and writes the summary tables, including the temperature spans.

```bash
T=datasets/raw/tches20/TCHES20V3_CNN_SCA
for i in 0 1 2 3 4 5; do
  python -m epi_comparison.epi_head_to_head --dataset-index $i \
    --dataset-dir $T/datasets --tches20-src-dir $T/src --models-dir $T/models \
    --output-dir results/epi_comparison/main
done
python -m epi_comparison.summarize_epi_head_to_head --input-dir results/epi_comparison/main --output-dir results/epi_comparison/summary
```

The EPI estimator itself (`fit_beta`, `information_records`) is the same as in
`paper_results/matched_metrics.py`, which computes the EPI column of Table 7
on the attack partitions.
