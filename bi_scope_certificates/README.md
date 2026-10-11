# bi_scope_certificates

A suite certificate covers only the suite's models. The local extension `BI^loc(epsilon)` also covers every attacker whose normalized scores stay within `epsilon` of a suite member:

```text
sup_y ||bar_s_h(y) - bar_s_center(y)||_infty <= epsilon
```

Such an attacker succeeds whenever the member's margin is at least `-2 epsilon`, so the certificate is again a count of attack traces. This folder computes those counts, plus the simpler scopes `BI^sing` (one frozen attacker) and `BI^suite`.

## Files

| File | Purpose | Used for |
|---|---|---|
| `compute_full_suite_bi_loc.py` | relaxed-margin counts of all 80 TCHES20 models of one benchmark over the nine radii | Table 9 |
| `profile_epsilon_calibration.py` | relaxed-margin counts of the twelve reference models on a 20% calibration and 80% certification split, plus their score variation under re-runs, batch sizes, FP16, quantization and noise | Figure 2 |
| `compute_bi_loc_margin_curves.py` | `BI^loc` curves of the reference models on their full attack sets | Section 5.3 |
| `prepare_aesrd_estranet_h5.py` | the AES-RD input file of the EstraNet reference model | Level C |
| `train_keras_ascad_random_key.py` | retrains the two Keras reference models of ASCAD random key (optional) | retraining |
| `build_bi_scope_tables.py` | `BI^sing` and `BI^loc` tables | full version |

## Run

Needs `requirements-ml.txt`, the TCHES20 benchmark, and `results/paper_results/bi_suite_dataset_summary.csv` (written by any level of `paper_results/`). The Transformer models also need the EstraNet sources and checkpoints of Level C.

The 80 ASCAD desync-50 models over the radius grid, then its two public reference models:

```bash
python -m bi_scope_certificates.compute_full_suite_bi_loc --dataset ascad_desync_50 \
  --suite-csv results/paper_results/bi_suite_dataset_summary.csv \
  --output-dir results/bi_scope_certificates/full_suite_loc

python -m bi_scope_certificates.profile_epsilon_calibration --dataset ascad_desync_50 --public-only \
  --output-dir results/bi_scope_certificates/calibration/ascad_desync_50
```

`paper_results/figure2_biloc_selected_centers.py` picks the calibrated radius on the 20% split (the first grid radius that changes the count of at least 75% of the reference models and covers their score variation) and certifies on the 80% split.

Retraining the two Keras models (written to `checkpoints/keras_ascad_random_key_retrained/`; pass that folder as `--trained-models-dir` to use them):

```bash
python -m bi_scope_certificates.train_keras_ascad_random_key --arch MLP
python -m bi_scope_certificates.train_keras_ascad_random_key --arch CNN
```

## Outputs

| File | Content |
|---|---|
| `<dataset>_full_suite_bi_loc_summary.csv` | per radius: largest count over the 80 models, its rate, the bound in bits, and the model that attains it |
| `<dataset>_full_suite_bi_loc_per_center.csv`, `_tie_audit.csv`, `_manifest.json` | per-model counts, ties and run settings |
| `epsilon_margin_counts.csv` | counts on the calibration and certification splits for every radius |
| `epsilon_variant_distances.csv` | largest score change per model and perturbation |
| `bi_loc_radius_curves.csv`, `BI_LOC_MARGIN_AUDIT_SUMMARY.md` | the curves and their summary |
