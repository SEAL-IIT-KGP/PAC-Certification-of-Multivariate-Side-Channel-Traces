# bi_scope_certificates

**What this folder does.** A suite certificate covers only the suite. The local extension `BI^loc(epsilon)` also covers every attacker whose normalized score function stays within an `epsilon` distance of a suite member:

```text
sup_y ||bar_s_h(y) - bar_s_center(y)||_infty <= epsilon
```

Such an attacker succeeds on a trace whenever the center's margin `rho_center >= -2 epsilon`, so the certificate is again a count: the attack traces with relaxed margin at least `-2 epsilon`. This folder computes those counts, plus the two simpler scopes `BI^sing` (one frozen attacker) and `BI^suite` (read from the suite summary).

## Files

| File | Purpose | Used for |
|---|---|---|
| `compute_full_suite_bi_loc.py` | relaxed-margin counts of all 80 TCHES20 models of one benchmark over the nine radii, with `delta` split over the models and the grid; writes per-center, summary and tie-audit CSVs plus a manifest | Table 9 |
| `profile_epsilon_calibration.py` | for the twelve reference models: splits each attack set once (seed 20260721) into 20% calibration and 80% certification traces, writes the relaxed-margin counts of both for every radius, and measures how far the scores move under repeated FP32 inference, other batch sizes, FP16 storage, quantization and input noise; `--public-only` keeps the TCHES20 models only | Figure 2 and its radius |
| `compute_bi_loc_margin_curves.py` | `BI^loc` curves of selected reference models (TCHES20 Keras MLP/CNN and EstraNet Transformers) on their attack records, one center per row (`M = 1`) | the bounds quoted at `eps = 1e-4` in Section 5.3 |
| `prepare_aesrd_estranet_h5.py` | converts the TCHES20 AES-RD arrays into the HDF5 file read by the EstraNet AES-RD reference model | Level C |
| `build_bi_scope_tables.py` | postprocessing: the `BI^sing` table from the suite summary and one `BI^loc` row per center selected with `--preferred-eps` | full version |

## Before you run

- `requirements-ml.txt` (TensorFlow); the EstraNet sources and checkpoints for the Transformer centers.
- The TCHES20 benchmark (`paper_results/fetch_tches20.sh`).
- The suite summary `results/paper_results/bi_suite_dataset_summary.csv`, written by the Table 7 builder at any level of `paper_results/`.

## Run

```bash
python -m bi_scope_certificates.compute_full_suite_bi_loc --help
python -m bi_scope_certificates.profile_epsilon_calibration --help
python -m bi_scope_certificates.compute_bi_loc_margin_curves --help
python -m bi_scope_certificates.build_bi_scope_tables --help
```

The 80 ASCAD d50 models over the radius grid (`--max-attack` defaults to the paper's attack-set sizes), then the two public reference models of the same benchmark with the 20%/80% split:

```bash
python -m bi_scope_certificates.compute_full_suite_bi_loc --dataset ascad_desync_50 \
  --suite-csv results/paper_results/bi_suite_dataset_summary.csv \
  --output-dir results/bi_scope_certificates/full_suite_loc

python -m bi_scope_certificates.profile_epsilon_calibration --dataset ascad_desync_50 --public-only \
  --output-dir results/bi_scope_certificates/calibration/ascad_desync_50
```

## Outputs

| File | Content |
|---|---|
| `<dataset>_full_suite_bi_loc_summary.csv` | per radius: the largest relaxed count over the 80 models, its rate, the certified bits, and the architecture that attains it |
| `<dataset>_full_suite_bi_loc_per_center.csv`, `_tie_audit.csv`, `_manifest.json` | per-model counts, ties, and the run settings |
| `epsilon_margin_counts.csv` | relaxed counts on the calibration and certification splits for every radius |
| `epsilon_variant_distances.csv` | the largest score change per model and perturbation |
| `bi_loc_radius_curves.csv`, `BI_LOC_MARGIN_AUDIT_SUMMARY.md` | the curves and their summary |

## Notes

- The calibrated radius of the paper comes from `profile_epsilon_calibration.py`: `paper_results/figure2_biloc_selected_centers.py` applies the rule to the 20% split (the first grid radius that changes the relaxed-margin count of at least 75% of the reference models and covers the observed FP32 score variation) and certifies on the 80% split.
- On CPU, repeated inference and other batch sizes move the scores by less than `1e-6`. The archived GPU profile of the Transformers reaches `1.9e-4`, which is what sets the printed radius `3e-4`.
- `compute_bi_loc_margin_curves.py` and `compute_full_suite_bi_loc.py` certify on all attack records they load (optionally the first `--max-attack`) over the full grid. Write outputs under `results/`.
