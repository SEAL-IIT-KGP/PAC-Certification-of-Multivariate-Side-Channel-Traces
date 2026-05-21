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

Use `datasets/README.md` for dataset path layout and write generated outputs under `results/`.
