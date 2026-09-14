# Dimension Stability Diagnostics

This directory contains the code for the paper diagnostics comparing BI stability with PI/HI/MI-style estimators as trace dimension grows.

- `multivariate_stability_driver.py`: synthetic multivariate stability driver.
- `real_nonbi_dimcap_baselines.py`: reduced-dimension non-BI baselines on real datasets.
- `real_projected_bi_dimcap.py`: projected-dimension BI certificates on real datasets.
- `ches_nonbi_metric_cell.py`: CHES-CTF high-dimensional non-BI metric cells.

Typical entry points:

```bash
python -m dimension_stability.multivariate_stability_driver --help
python -m dimension_stability.real_nonbi_dimcap_baselines --help
python -m dimension_stability.real_projected_bi_dimcap --help
python -m dimension_stability.ches_nonbi_metric_cell --help
```

Notes:

- `multivariate_stability_driver.py`: `--include-baselines/--no-include-baselines` (`--no-baselines` also disables them) and `--resume/--no-resume` both default to on. With `--sweep-dimensions`, the figures are the pooled per-dimension coefficient-of-variation and success-rate plots. The per-dataset bar chart is produced only by non-sweep runs with more than one dataset.
- `ches_nonbi_metric_cell.py`: a cell is skipped only if the checkpoint already has a `status=complete` row for it. Failed cells are re-run. Pass `--force` to re-run completed cells too.

Use `datasets/README.md` for dataset path layout and write generated outputs under `results/`.
