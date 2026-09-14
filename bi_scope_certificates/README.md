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
- `build_bi_scope_tables.py`: postprocessing only. It builds the `BI^sing` table from the suite summary
  CSV and selects one `BI^loc` row per center from the radius curves using `--preferred-eps`.

The epsilon-calibration split described in the paper's appendix (20% calibration / 80% certification of
the reserved attack traces) is not implemented in these scripts. They certify on the attack records they
load (optionally truncated to the first `max_attack` records) over the full epsilon grid, and they do not
select `epsilon_cal`.

Main entry points:

```bash
python -m bi_scope_certificates.compute_bi_loc_margin_curves --help
python -m bi_scope_certificates.compute_full_suite_bi_loc --help
python -m bi_scope_certificates.build_bi_scope_tables --help
```

Outputs should be written under `results/bi_scope_certificates/` or another ignored output path.
