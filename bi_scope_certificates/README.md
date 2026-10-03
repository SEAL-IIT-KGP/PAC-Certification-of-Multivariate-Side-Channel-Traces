# BI Scope Certificates

This directory contains scripts for three declared attacker scopes:

- `BI^sing`: a singleton certificate for one frozen attacker per dataset.
- `BI^suite`: the finite frozen-suite certificate, read from the suite summary CSV.
- `BI^loc(epsilon)`: score-local families around frozen reference models, evaluated by relaxed
  margin counts.

The local family is defined directly in normalized score space:

```text
sup_y ||bar_s_h(y) - bar_s_center(y)||_infty <= epsilon
```

and an attack trace counts as a relaxed success when `rho_center >= -2 epsilon`.

Scripts:

- `compute_bi_loc_margin_curves.py`: evaluates selected frozen reference models (TCHES20 Keras MLP/CNN
  models and EstraNet Transformer checkpoints) on their attack records and writes `BI^loc` curves over
  the epsilon grid to `bi_loc_radius_curves.csv`, with one center per row (`M=1`).
- `compute_full_suite_bi_loc.py`: computes the same relaxed-margin curves for all 80 TCHES20 pretrained
  models of one dataset, splitting `delta` over the evaluated models and the epsilon grid. It writes
  per-center, summary, and tie-audit CSVs plus a manifest.
- `profile_epsilon_calibration.py`: for the twelve selected reference models, splits each attack set
  once (seed 20260721) into 20% calibration and 80% certification traces, writes the relaxed-margin counts
  of both splits for every radius, and measures how far the normalized scores move under repeated FP32
  inference, other batch sizes, FP16 storage, quantization, and input noise. `--public-only` restricts a
  run to the public TCHES20 reference models.
- `prepare_aesrd_estranet_h5.py`: converts the TCHES20 AES-RD arrays into the HDF5 file read by the
  EstraNet AES-RD reference model.
- `build_bi_scope_tables.py`: postprocessing only. It builds the `BI^sing` table from the suite summary
  CSV and selects one `BI^loc` row per center from the radius curves using `--preferred-eps`.

`compute_bi_loc_margin_curves.py` and `compute_full_suite_bi_loc.py` certify on all attack records they
load (optionally truncated to the first `max_attack` records) over the full epsilon grid. The calibrated
radius of the paper comes from `profile_epsilon_calibration.py`: `paper_results/figure2_biloc_selected_centers.py`
applies the calibration rule to the 20% split (the first grid radius that changes the relaxed-margin count
of at least 75% of the reference models and covers the observed FP32 score variation) and certifies on the
80% split.

Main entry points:

```bash
python -m bi_scope_certificates.compute_bi_loc_margin_curves --help
python -m bi_scope_certificates.compute_full_suite_bi_loc --help
python -m bi_scope_certificates.profile_epsilon_calibration --help
python -m bi_scope_certificates.build_bi_scope_tables --help
```

Outputs should be written under `results/bi_scope_certificates/` or another ignored output path.
